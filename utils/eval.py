from __future__ import print_function, absolute_import
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp
import time
import datetime
from .misc import AverageMeter
from .mixup import mixup_data, mixup_criterion
from .progress.progress.bar import Bar as Bar
# for zca whitening
from .zca import ZCATransformation
import scipy.io as sio

__all__ = ['accuracy', 'accuracy_mixup', 'train', 'test']

#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True

def accuracy(output, target, topk=(1,), num_classes=10):
    """Computes the precision@k for the specified values of k"""
    if num_classes < 5:
        topk = (1, )
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0).double()
            res.append(correct_k.mul_(100.0 / batch_size))
        if len(topk) == 1:
            res.append(correct_k.mul(0))
        return res


def accuracy_mixup(output, targets_a, targets_b, lam, topk=(1,), num_classes=10):
    """Computes the precision@k for the specified values of k"""
    if num_classes < 5:
        topk = (1, )
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets_a.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (lam * pred.eq(targets_a.view(1, -1).expand_as(pred))
			+ (1 - lam) * pred.eq(targets_b.view(1, -1).expand_as(pred)))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0).double()
            res.append(correct_k.mul_(100.0 / batch_size))
        if len(topk) == 1:
            res.append(correct_k.mul(0))
        return res

def train(train_loader, model, teacher, criterion, optimizer, scheduler, scaler, epoch, writer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_ce = AverageMeter()
    losses_kd = AverageMeter()
    losses_at = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    # print('Count of using GPUs:', torch.cuda.device_count())
    # print('Current cuda device:', torch.cuda.current_device())
    if args.transfer_mode != 0:
        teacher.eval()
        teacher.cuda()

    if args.transfer_mode >= 2 and args.transfer_mode <= 3:
        teacher.module.transfer = True
        model.module.transfer = True

    if args.dali:
        assert args.dataset == 'imagenet', 'Currently, dali loader is used only for ImageNet dataset.'
        len_trainloader = int(train_loader._size/train_loader.batch_size)+1
    else:
        len_trainloader = len(train_loader)

    if args.rank == 0:
        bar = Bar('Processing', max=len_trainloader)

    if args.zca:
        mat_contents = sio.loadmat('./utils/zca_cifar10.mat')
        transformation_matrix = torch.from_numpy(mat_contents['zca_matrix']).float()
        transformation_mean = torch.from_numpy(mat_contents['zca_mean'][0]).float()
        zca = ZCATransformation(transformation_matrix, transformation_mean)

    for batch_idx, data in enumerate(train_loader):
        if args.dali:
            inputs = data[0]["data"]
            targets = data[0]["label"].squeeze().long()
        else:
            inputs, targets = data

        if args.zca:
            inputs = zca(inputs)

        # measure data loading time
        data_time.update(time.time() - end)

        # when to set non_blocking=True
        # https://discuss.pytorch.org/t/why-set-cuda-non-blocking-false-for-target-variables/16943
        # Differences between .to(), .cuda()
        # https://stackoverflow.com/questions/53331247/pytorch-0-4-0-there-are-three-ways-to-create-tensors-on-cuda-device-is-there-s
        inputs = inputs.to("cuda")
        targets = targets.to("cuda", non_blocking=True)


        if args.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha)

        # tensorboard add_graph.
        if batch_idx == 0 and epoch == 0:
            writer.add_graph(model, inputs)
        # tensorboard add_hooks.
        if batch_idx == 0:
            writer.add_hooks(model, epoch)


        # Runs the forward pass with autocasting.
        if scaler is not None:
            with torch.cuda.amp.autocast(enabled=args.amp):
                # compute output
                if args.transfer_mode == 0:
                    outputs = model(inputs)
                elif (args.transfer_mode == 1) or (args.transfer_mode == 4):
                    outputs = model(inputs)
                    with torch.no_grad():
                        outputs_teacher = teacher(inputs)
                elif args.transfer_mode >= 2:
                    outputs, feats = model(inputs)
                    with torch.no_grad():
                        outputs_teacher, feats_teacher = teacher(inputs)

                # compute original loss
                if args.mixup:
                    loss_ce = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss_ce = criterion(outputs, targets)
                # compute teacher-student loss
                if args.transfer_mode == 1:
                    loss_kd = kd_loss_fn(outputs, outputs_teacher, args.kd_temperature, args.kd_alpha)
                    loss_ce = loss_ce * (1. - args.kd_alpha)
                    loss_at = 0
                elif args.transfer_mode == 2:
                    loss_at = sum([at_loss(x,y) for x, y in zip(feats, feats_teacher)]) * args.at_beta
                    loss_kd = 0
                elif args.transfer_mode == 3:
                    loss_kd = kd_loss_fn(outputs, outputs_teacher, args.kd_temperature, args.kd_alpha)
                    loss_at = sum([at_loss(x,y) for x, y in zip(feats, feats_teacher)]) * args.at_beta
                    loss_ce = loss_ce * (1. - args.kd_alpha)
                elif args.transfer_mode == 4: #eh code kd loss
                    loss_kd = -1 * torch.mean(
                            torch.sum(F.softmax(outputs_teacher, dim=1) * F.log_softmax(outputs, dim=1), dim=1))
                    loss_at = 0
                else:
                    loss_kd = 0
                    loss_at = 0

                # penalty loss
                # loss_pen = F.huber_loss()

                loss = loss_ce + loss_kd + loss_at
        else:
            # compute output
            if args.transfer_mode == 0:
                outputs = model(inputs)
            elif (args.transfer_mode == 1) or (args.transfer_mode == 4):
                outputs = model(inputs)
                with torch.no_grad():
                    outputs_teacher = teacher(inputs)
            elif args.transfer_mode >= 2:
                outputs, feats = model(inputs)
                with torch.no_grad():
                    outputs_teacher, feats_teacher = teacher(inputs)

            # compute original loss
            if args.mixup:
                loss_ce = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss_ce = criterion(outputs, targets)

            # compute teacher-student loss
            if args.transfer_mode == 1:
                loss_kd = kd_loss_fn(outputs, outputs_teacher, args.kd_temperature, args.kd_alpha)
                loss_ce = loss_ce * (1. - args.kd_alpha)
                loss_at = 0
            elif args.transfer_mode == 2:
                loss_at = sum([at_loss(x,y) for x, y in zip(feats, feats_teacher)]) * args.at_beta
                loss_kd = 0
            elif args.transfer_mode == 3:
                loss_kd = kd_loss_fn(outputs, outputs_teacher, args.kd_temperature, args.kd_alpha)
                loss_at = sum([at_loss(x,y) for x, y in zip(feats, feats_teacher)]) * args.at_beta
                loss_ce = loss_ce * (1. - args.kd_alpha)
            elif args.transfer_mode == 4:
                loss_kd = -1 * torch.mean(
                        torch.sum(F.softmax(outputs_teacher, dim=1) * F.log_softmax(outputs, dim=1), dim=1))
                loss_at = 0
            else:
                loss_kd = 0
                loss_at = 0

            loss = loss_ce + loss_kd + loss_at

        # measure accuracy and record loss
        if args.mixup:
            prec1, prec5 = accuracy_mixup(outputs.data, targets_a.data, targets_b.data, lam, topk=(1,5), num_classes=args.num_classes)
        else:
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1,5), num_classes=args.num_classes)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # Backpropagation
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # if 'binary' in args.arch:
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             if ('pconv' in name) or ('plinear' in name):
        #                 print(name, param[0], param.grad[0])
        #                 import pdb; pdb.set_trace()

        # Loss are not averaged in DDP: they are kept individually, BUT gradients are reduced during loss.backward().
        # But we average the loss for all gpus and print them.
        if args.distributed:
            loss_ce = reduce_tensor(loss_ce.data, args)
            prec1 = reduce_tensor(prec1, args)
            prec5 = reduce_tensor(prec5, args)
            if (args.transfer_mode > 0) and (args.transfer_mode != 2):
                loss_kd = reduce_tensor(loss_kd.data, args)
            if (args.transfer_mode >= 2) and (args.transfer_mode <= 3):
                loss_at = reduce_tensor(loss_at.data, args)

        losses_ce.update(loss_ce.item(), inputs.size(0))
        if (args.transfer_mode > 0) and (args.transfer_mode != 2):
            losses_kd.update(loss_kd.item(), inputs.size(0))
        if (args.transfer_mode >= 2) and (args.transfer_mode <= 3):
            losses_at.update(loss_at.item(), inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if batch_idx == 0:
            writer.remove_hooks()
            writer.log_grads(model, epoch)

        # scheduler.step(epoch + (batch_idx+1) / len_trainloader) # is it ok if it disappear?

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if args.rank == 0:
            # bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_ce: {loss_ce:.4f} | Loss_kd: {loss_kd:.4f} | Loss_at: {loss_at:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | Loss_ce: {loss:.4f} | Loss_kd: {loss_kd:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len_trainloader,
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                loss=losses_ce.avg,
                # loss_ce=losses_ce.avg,
                loss_kd=losses_kd.avg,
                # loss_at=losses_at.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
    if args.rank == 0:
        bar.finish()

    return (losses_ce.avg+losses_kd.avg+losses_at.avg, top1.avg, top5.avg)

def test(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    if args.dali:
        len_valloader = int(val_loader._size/val_loader.batch_size)
    else:
        len_valloader = len(val_loader)

    if args.rank == 0:
        bar = Bar('Processing', max=len_valloader)

    if args.zca:
        mat_contents = sio.loadmat('./utils/zca_cifar10.mat')
        transformation_matrix = torch.from_numpy(mat_contents['zca_matrix']).float()
        transformation_mean = torch.from_numpy(mat_contents['zca_mean'][0]).float()
        zca = ZCATransformation(transformation_matrix, transformation_mean)


    with torch.no_grad():
        start_time=time.time()
        for batch_idx, data in enumerate(val_loader):
            if args.dali:
                inputs = data[0]["data"]
                targets = data[0]["label"].squeeze().long()
            else:
                inputs, targets = data

            # measure data loading time
            data_time.update(time.time() - end)

            if args.zca:
                inputs = zca(inputs)

            inputs = inputs.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)

            # compute output
            model.module.transfer = False
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # if 'binary' in args.arch:
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             if ('pconv' in name) or ('plinear' in name):
            #                 print(name, param[0], param.grad[0])
            #                 import pdb; pdb.set_trace()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5), num_classes=args.num_classes)
            # Loss are not averaged in DDP: they are kept individually, BUT gradients are reduced during loss.backward().
            # But we average the loss for all gpus and print them.
            if args.distributed:
                loss = reduce_tensor(loss.data, args)
                prec1 = reduce_tensor(prec1, args)
                prec5 = reduce_tensor(prec5, args)

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if args.rank == 0:
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len_valloader,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
                bar.next()

            if args.whist:
                print("\nSave weight hist so exit")
                exit()

        if args.rank == 0:
            print('\nEvaluation Total Time: {total_time}s'.format(total_time=str(datetime.timedelta(seconds=(time.time()-start_time)))))
            bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def log_test(meas_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    if args.dali:
        len_measloader = int(meas_loader._size/meas_loader.batch_size)
    else:
        len_measloader = len(meas_loader)

    if args.rank == 0:
        bar = Bar('Processing', max=len_measloader)

    with torch.no_grad():
        start_time=time.time()
        for batch_idx, data in enumerate(meas_loader):
            if args.dali:
                inputs = data[0]["data"]
                targets = data[0]["label"].squeeze().long()
            else:
                inputs, targets = data

            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)

            # compute output
            model.module.transfer = False
            outputs = model(inputs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if args.rank == 0:
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} |'.format(
                    batch=batch_idx + 1,
                    size=len_measloader,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                )
                bar.next()

        print('\nParameter Search Time: {total_time}s'.format(total_time=time.time()-start_time))

        if args.rank == 0:
            bar.finish()
    return


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    # This is how at is implemented in Larq implementation.
    return (at(x) - at(y)).pow(2).mean()

    # This is what the original at paper describes.
    # return (at(x) - at(y)).pow(2).sum().sqrt()


def kd_loss_fn(outputs, teacher_outputs, temp, alpha):
    """Compute the knowledge-distillation (KD) loss given outputs, labels.

    Note: the KL Divergence in Pytorch expects the input tensor to be log probabilities.
    """
    KD_loss = F.kl_div(F.log_softmax(outputs/temp, dim=1), F.softmax(teacher_outputs / temp, dim=1),
                       reduction="batchmean") * alpha * (temp ** 2)
        # .log_softmax(outputs/args.kd_temperature, dim=1),
        #                      F.log_softmax(teacher_outputs/args.kd_temperature, dim=1)) * (args.kd_alpha * args.kd_temperature * \
        #                      args.kd_temperature)

    return KD_loss
