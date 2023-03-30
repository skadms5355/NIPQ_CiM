from __future__ import print_function

import os
import time
import random
import warnings
import pathlib
import numpy as np
import pandas as pd
import socket
from datetime import datetime

import models
import argparse
from utils.arguments import set_arguments, check_arguments

args = set_arguments()

from utils import data_loader, initialize, logging, misc, tensorboard, eval
from utils.misc import ForkedPdb
from utils.schedule_train import set_optimizer, set_scheduler

from utils.preproc_bi_real_net import CrossEntropyLabelSmooth	# for Bi-real-net
from models.psum_modules import get_statistics_from_hist, set_bitserial_layer, set_BitSerial_log, set_Noise_injection, unset_BitSerial_log
from models.nipq_quantization_module import bops_cal

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Possibly corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Metadata Warning", UserWarning)

# import torch modules
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
assert torch.cuda.is_available(), "check if cuda is avaliable"


def PickUnusedPort():
    '''Picks unused port in the current node'''
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    addr, port = s.getsockname()
    s.close()

    return port


def SetRandomSeed(seed=None):
    '''Sets random seed for all torch methods'''
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def main():
    check_arguments(args)

    # it is required that the number of gpu's visible by each node should be same.
    # we don't know if setting them using gpu-id work yet.
    ngpus_per_node = torch.cuda.device_count()

    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    # nr == initial rank, nodes == initial world_size in above code.
    # In the example above (1 node, multigpu), world_size: 1, distributed: True,
    # multiprocessing_distributed: True,
    args.nr = 0
    args.nodes = 1
    args.world_size = 1

    if args.distributed:
        # world size equals the total available gpus for DDP, fixed to 1 in DP.
        args.world_size = ngpus_per_node * args.nodes
        args.dist_backend = "nccl"
        args.dist_url = f"tcp://127.0.0.1:{PickUnusedPort()}"
        print(f"Binding to processes using : {args.dist_url}")

    if not args.psum_comp:
        assert False, f"This script only supports psum quantization"

    if args.checkpoint is None:
        prefix = os.path.join("checkpoints", args.dataset, args.model_mode, args.arch)

        if args.evaluate:
            prefix = os.path.join(prefix, "eval")

        if args.model_mode == 'nipq':
            prefix = os.path.join(prefix, "{}_fix:{}".format(args.nipq_noise, args.fixed_bit))
        elif args.model_mode == 'hn_quant':
            prefix = os.path.join(prefix, "a:{}_w:{}".format(args.abits, args.wbits), "trained_noise_{}_ratio_{}".format(args.trained_noise, args.ratio))
        else:
            prefix = os.path.join(prefix, "a:{}_w:{}".format(args.abits,args.wbits))
        
        if args.mapping_mode != 'none':

            if args.arraySize > 0 and args.psum_comp: # inference with psum comp
                prefix = os.path.join(prefix, args.mapping_mode, "{}_c:{}_{}".format(str(args.arraySize), args.cbits, args.per_class))
            else:
                prefix = os.path.join(prefix, args.mapping_mode, "no_psum_c:{}".format(args.cbits))

            if args.is_noise:
                if args.noise_type == 'meas':
                    type = '{}'.format(args.meas_type)
                else:
                    type = 'type_{}'.format(args.co_noise)

                if args.tn_file is not None:
                    prefix = os.path.join(prefix, '{}_{}_{}').format(args.tn_file, args.noise_type, type)
                else:
                    prefix = os.path.join(prefix, '{}_{}').format(args.noise_type, type)
        else:
            pass

        if args.local is None:
            prefix = prefix
        else:
            prefix_bak = prefix
            prefix = os.path.join(args.local, prefix_bak)

        if args.psum_comp:
            args.checkpoint = os.path.join(prefix, "log_bitserial_info")
            if not os.path.exists(args.checkpoint):
                os.makedirs(args.checkpoint)
                os.makedirs(os.path.join(args.checkpoint, 'hist'))
            else:
                if args.log_file:
                    print(f"remove folder {args.checkpoint}")
                    os.system(f'rm -rf {args.checkpoint}')
                    print(f"create new folder {args.checkpoint}")
                    os.makedirs(args.checkpoint)
                    os.makedirs(os.path.join(args.checkpoint, 'hist'))
        else:
            args.checkpoint = misc.mkdir_now(prefix)
    else: 
        os.makedirs(args.checkpoint, exist_ok=True)
    print(f"==> Save everything in \n {args.checkpoint}")

    misc.lndir_p(args.checkpoint, args.link)

    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args), join=True)
    else:
        main_worker(0, ngpus_per_node, args)

    # move to checkpoint file
    if args.local:
        dest_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), prefix_bak)
        misc.copy_folder(args.checkpoint, dest_path)


# gpu numbers vary from 0 to (ngpus_per_node - 1)
def main_worker(gpu, ngpus_per_node, args):

    # initialize parameters for results
    top1 = {'train': 0, 'valid': 0, 'test': 0, 'best_valid': 0, 'test_at_best_valid_top1': 0, 'best_test': 0}
    top5 = {'train': 0, 'valid': 0, 'test': 0, 'best_valid': 0, 'test_at_best_valid_top1': 0, 'best_test': 0}
    loss = {'train': None, 'valid': None, 'test': None}

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Sets current device ordinal.
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    # TODO (VINN): when implementing multi-node DDP, check these values.
    # Check https://github.com/pytorch/examples/blob/master/imagenet/main.py
    if args.distributed:
        # for nodes = 2, ngpus_per_node = 4,
        # rank = 0 ~ 3 for gpus in nr 0, 4 ~ 7 for gpus in nr 1
        args.rank = args.nr * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0

    args.manualSeed = SetRandomSeed(seed=args.manualSeed)
    writer = tensorboard.Tensorboard_writer(args.checkpoint, args.tensorboard, args.rank)

    # Set ddp arguments
    if args.distributed:
        args.train_batch = int(args.train_batch / ngpus_per_node)
        args.test_batch = int(args.test_batch / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        print(f"gpu: {args.gpu}, current_device: {torch.cuda.current_device()}, "
              f"train_batch_per_gpu: {args.train_batch}, workers_per_gpu: {args.workers}")
    else:
        print(f"gpu: {args.gpu_id}, current_device: {torch.cuda.current_device()}, "
              f"train_batch_total: {args.train_batch}, workers_total: {args.workers}")


    # Set data loader
    cudnn.benchmark = True
    #cudnn.deterministic = True
    train_loader, train_sampler, valid_loader, test_loader = data_loader.set_data_loader(args)

    # Load teacher model if required
    if args.teacher is not None:
        print("==> Loading teacher model...")
        assert os.path.exists(args.teacher), 'Error: no teacher model found!'
        state = torch.load(args.teacher, map_location=lambda storage, loc: storage.cuda(args.gpu))
        teacher = models.__dict__[state['args_dict']['arch']](**state['args_dict'])
#        for p in teacher.parameters():
#            p.requires_grad = False
    else:
        teacher = None

    # Create model.
    assert torch.backends.cudnn.enabled, 'Amp requires cudnn backend to be enabled.'
    args_dict = vars(args)
    model = models.__dict__[args.arch](**args_dict)

        
    # initialize weights
    modules_to_init = ['Conv2d', 'BinConv', 'Linear', 'BinLinear', \
            'QConv', 'QLinear', 'QuantConv', 'QuantLinear', \
            'PsumQConv', 'PsumQLinear']
    bn_modules_to_init = ['BatchNorm1d', 'BatchNorm2d']
    for m in model.modules():
        if type(m).__name__ in modules_to_init:
            initialize.init_weight(m, method=args.init_method, dist=args.init_dist, mode=args.init_fan)
            # print('Layer {} has been initialized.'.format(type(m).__name__))
        if type(m).__name__ in bn_modules_to_init:
            if args.bn_bias != 0.0:    
                # print(f"Initializing BN bias to {args.bn_bias}")
                torch.nn.init.constant_(m.bias, args.bn_bias)
                #print('Layer {} has been initialized.'.format(type(m).__name__))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
 
    # We want to make sure that we apply weight decay only to weights.
    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        #print(f"name: {pname}, dimension: {p.ndimension()}")
        if p.ndimension() != 1 and 'weight' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = set_optimizer(model, other_parameters, weight_parameters, args)
    
    # Initialize scaler if amp is used.
    if args.amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    else:
        scaler = None

    # Initialize scheduler. This is done after the amp initialize to avoid warning.
    # Refer to Issues: https://discuss.pytorch.org/t/cyclic-learning-rate-how-to-use/53796
    scheduler, after_scheduler = set_scheduler(optimizer, args)

    # Wrap the model with DDP
    if args.distributed:
        print(f"student gpu-id: {args.gpu}")
        model = torch.nn.parallel.DistributedDataParallel(
            model.to("cuda"), device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).to("cuda")


    # Wrap the teacher model with DDP if exist
    if teacher is not None:
        if args.distributed:
            print(f"teacher gpu-id: {args.gpu}")
            teacher = torch.nn.parallel.DistributedDataParallel(
                teacher.to("cuda"), device_ids=[args.gpu], output_device=args.gpu)
        else:
            teacher = torch.nn.DataParallel(teacher).to("cuda")
        teacher.load_state_dict(state['state_dict'])

    # Print model information
    if args.rank == 0:
        print('---------- args -----------')
        print(args)
        print('---------- model ----------')
        print(model)

        ## TODO (VINN): We want to take quantization into account!
        total_parameters = sum(p.numel() for p in model.parameters())/1000000.0
        print(f"    Total params: {total_parameters:.2f}M")


    # Load pre-trained model if enabled
    if args.pretrained:
        if args.rank == 0:
            print('==> Get pre-trained model..')
        assert os.path.exists(args.pretrained), 'Error: no pre-trained model found!'

        load_dict = torch.load(args.pretrained, map_location=lambda storage, loc: storage.cuda(args.gpu))['state_dict']
        model_dict = model.state_dict()
        model_keys = model_dict.keys()
        for name, param in load_dict.items():
            if name in model_keys:
                model_dict[name] = param
        model.load_state_dict(model_dict)
        
        if not args.psum_comp and test_loader is not None:
            loss['test'], top1['test'], top5['test'] = eval.test(test_loader, model, criterion, 0, args)
            if args.rank == 0:
                print(f"Test loss: {loss['test']:<10.6f} Test top1: {top1['test']:<7.4f} Test top5: {top5['test']:<7.4f}")

    # Overwrite arguments from reading checkpoint's arg_dict
    # Load from checkpoint if resume is enabled.
    if args.resume:
        if args.rank == 0:
            print('==> Overwriting arguments ... (Resuming)')
        assert os.path.isdir(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(args.resume, 'checkpoint.pth.tar'),
                                map_location=lambda storage, loc: storage.cuda(args.gpu))
        # overwrite, except for special arguments.
        args_dict = checkpoint['args_dict']
        assert args_dict['gpu_id'].count(",") == args.gpu_id.count(","), \
                'Do not change the number of GPU used when resuming!'
        args_dict['gpu_id'] = args.gpu_id
        args_dict['workers'] = args.workers
        args_dict['resume'] = args.resume
        args_dict['gpu'] = args.gpu
        args_dict['rank'] = args.rank
        args_dict['nr'] = args.nr
        args_dict['checkpoint'] = args.checkpoint
        args = argparse.Namespace(**args_dict)

        if args.rank == 0:
            print('------ resuming with args -------')
            print(args)
            print('==> Resuming from checkpoint..')

        # Load checkpoint.
        top1 = checkpoint['top1']
        top5 = checkpoint['top5']
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        after_scheduler.load_state_dict(checkpoint['after_scheduler'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.amp:
            scaler.load_state_dict(checkpoint['amp'])

    if args.rank == 0:
        # Save model structure and arguments to model.txt inside the checkpoint.
        with open(f"{args.checkpoint}/model.txt", 'w') as f:
            f.write(str(model) + "\n")
            for key, value in args_dict.items():
                f.write(f"{key:<20}: {value}\n")


    # Resume and initializing logger
    title = args.dataset + '-' + args.arch

    # This script only support bitserial mode with arraySize > 0
    assert args.arraySize > 0, "This script only support arraySize > 0"

    logger_path = os.path.join(args.checkpoint, f'log_scan_pbits_{args.pbits}.pkl')
    best_path = os.path.join(args.checkpoint, f'best_scan.pkl')

    numlayers = 0
    if 'vgg9' in args.arch:
        numlayers = 7
    elif 'alexnet' in args.arch:
        numlayers = 6
    else:
        assert False, f"This script does not support architecture type {args.arch}"
    
    if args.rank % ngpus_per_node == 0:
        # Resume and initializing logger
        if args.evaluate:
            report_path = os.path.join(str(pathlib.Path().resolve()), ((args.checkpoint.replace('checkpoints', 'report')).replace('eval/', '')).replace('/log_bitserial_info', '').replace(f'type_{args.co_noise}', 'type'))
            graph_path = os.path.join(str(pathlib.Path().resolve()), 'graph', args.dataset, f'Psum_{args.arch}', args.mapping_mode, args.psum_mode, 'class_{}'.format(args.per_class))
            if not args.psum_comp:
                report_path = '/'.join(report_path.split('/')[:-1]) # time folder remove 
            os.makedirs(report_path, exist_ok=True)
            report_file = os.path.join(report_path, 'model_report.pkl')

    if args.evaluate:
        if args.rank == 0:
            print('\nEvaluation only')

        if args.model_mode == 'nipq':
            from models.nipq_hwnoise_psum_module import PsumQuantOps as PQ
            PQ.psum_initialize(model, act=True, weight=True, fixed_bit=args.fixed_bit, cbits=args.cbits, arraySize=args.arraySize, mapping_mode=args.mapping_mode, \
                                psum_mode=args.psum_mode, wbit_serial=args.wbit_serial, pbits=args.pbits, pclipmode=args.pclipmode, pclip=args.pclip, psigma=args.psigma, \
                                checkpoint=args.checkpoint, log_file=args.log_file)
            if args.is_noise and 'hwnoise' in args.nipq_noise:
                PQ.hwnoise_initilaize(model, weight=True, hwnoise=True, cbits=args.cbits, mapping_mode=args.mapping_mode, co_noise=args.co_noise, \
                                    noise_type=args.noise_type, res_val=args.res_val)
        elif (args.model_mode == 'quant') or (args.model_mode == 'hn_quant'):
            set_BitSerial_log(model, checkpoint=args.checkpoint, log_file=args.log_file,\
                pbits=args.pbits, pclipmode=args.pclipmode, pclip=args.pclip, psigma=args.psigma)
            if args.is_noise:
                set_Noise_injection(model, co_noise=args.co_noise, ratio=args.ratio)
        else:
            assert False, "This mode is not supported psum computation"

        # search layer-wise min, max, value 
        if args.log_file:

            if args.class_split:
                eval.log_test(valid_loader, model, args)
            else:
                eval.log_test(train_loader, model, args)

            if args.model_mode == 'nipq':
                PQ.unset_bitserial_log(model)
            elif (args.model_mode == 'quant') or (args.model_mode == 'hn_quant'):
                unset_BitSerial_log(model)

        start_time=time.time()
        # disable the bitserial operation for the entire network
        for idx in range(0, numlayers):
            if args.model_mode == 'nipq':
                PQ.set_bitserial_layer(model, idx, wbit_serial=False)
            elif (args.model_mode == 'quant') or (args.model_mode == 'hn_quant'):    
                set_bitserial_layer(model, idx, abit_serial=False, wbit_serial=False)

        print('reset the model before bitserial psum quant optimization')

        # optimize the psum quant iteratively
        for idx in range(0, numlayers):
            layer_time = time.time()
            if args.model_mode == 'nipq':
                PQ.set_bitserial_layer(model, idx, wbit_serial=args.wbit_serial)
            elif (args.model_mode == 'quant') or (args.model_mode == 'hn_quant'):
                set_bitserial_layer(model, idx, abit_serial=True, wbit_serial=args.wbit_serial)
            
            # get maxV, minV information for target layer
            filepath = f'{args.checkpoint}/hist/layer{idx}_hist.pkl'
            # if os.path.isfile(filepath):
            df = pd.read_pickle(filepath)
            mean, std, minbound, maxbound = get_statistics_from_hist(df)
            print(f'Max Value [{maxbound}] | Min Value [{minbound}]')

            if (args.mapping_mode == 'two_com') or (args.mapping_mode == 'ref_d'):
                maxpbound = int(2**args.pbits)-1 if maxbound < 2**args.pbits else maxbound
                minpbound = int(2**args.pbits)-2 if minbound < 2**args.pbits else minbound
                center = minbound
            else:
                maxpbound = max(maxbound, abs(minbound))
                minpbound = int(2**args.pbits)-2
                center = 0
            print(f'Search Range: Max Value [{maxpbound}] | Min Value [{minpbound}]')
                
            # pbound scan
            best_acc = 0
            best_pbound = 0
            for pbound in range(maxpbound, minpbound, -1):
                pbound_time=time.time()
                print(f"layer is {idx}, pbound is {pbound}")
                if args.model_mode == 'nipq':
                    PQ.set_bound_layer(model, pquant_idx=idx, pbits=args.pbits, pbound=pbound-center, center=center)
                else:
                    set_BitSerial_log(model, checkpoint=args.checkpoint,
                                    pbits=args.pbits, pclipmode=args.pclipmode, pclip=args.pclip, psigma=args.psigma,
                                    pquant_idx=idx, pbound=pbound-center, center=center)

                # do evaluation 
                epoch = 0 
                if valid_loader is not None:
                    loss['valid'], top1['valid'], top5['valid'] = eval.test(valid_loader, model, criterion, epoch, args)
                    if args.rank == 0:
                        print(f"Valid loss: {loss['valid']:<10.6f} Valid top1: {top1['valid']:<7.4f} Valid top5: {top5['valid']:<7.4f}")

                # update best pbound
                if best_acc <= top1['valid']:
                    if args.rank == 0:
                        print(f"Update best_acc: {top1['valid']}, best_pbound: {pbound} (old best_acc: {best_acc})")
                    best_acc = top1['valid']
                    best_pbound = pbound

                # log evaluation result
                if args.rank == 0:
                    if os.path.isfile(logger_path):
                        df = pd.read_pickle(logger_path)
                    else:
                        df = pd.DataFrame()

                    df = df.append({
                        "pbits":        args.pbits,
                        "pquant_idx":   idx,
                        "pbound":       pbound,
                        "Valid Top1":   top1['valid'],
                        }, ignore_index=True)
                    df.to_pickle(logger_path)

                if args.dali:
                    if valid_loader is not None:
                        valid_loader.reset()

            # update best pbound and get accuracy
            # set QuantPsum with given pbits and best pbound for next iteration
            if args.model_mode == 'nipq':
                PQ.set_bound_layer(model, pquant_idx=idx, pbits=args.pbits, pbound=best_pbound-center, center=center)
            else:
                set_BitSerial_log(model, checkpoint=args.checkpoint,
                                    pbits=args.pbits, pclipmode=args.pclipmode, pclip=args.pclip, psigma=args.psigma,
                                    pquant_idx=idx, pbound=best_pbound-center, center=center)
            
            # evaluate
            if args.rank == 0:
                print(f"Evaluate best pbound found on layer_idx: {idx}, pbits: {args.pbits}, best_pbound: {best_pbound}")
            loss['test'], top1['test'], top5['test'] = eval.test(test_loader, model, criterion, epoch, args)
            if args.rank == 0:
                print(f"Test loss: {loss['test']:<10.6f} Test top1: {top1['test']:<7.4f} Test top5: {top5['test']:<7.4f}")

            # log best evaluation result
            if args.rank == 0:
                if os.path.isfile(best_path):
                    df = pd.read_pickle(best_path)
                    df=df[(df.pquant_idx != idx) | (df.pbits != args.pbits)]
                else:
                    df = pd.DataFrame()
                # save best case
                df = df.append({
                    "pbits":            args.pbits,
                    "pquant_idx":       idx,
                    "best_pbound":      best_pbound,
                    "Test Top1":        top1['test'],
                    }, ignore_index=True)
                df.to_pickle(best_path)
                df.to_csv(args.checkpoint+'/scan_bound.txt', sep = '\t', index = False)

                # apply and evaluate the best pbound
                # update report

                if os.path.isfile(report_file):
                    df = pd.read_pickle(report_file)
                    if args.testlog_reset:
                        df=df[(df.pquant_idx != idx) | (df.pbits != args.pbits)]
                else:
                    df = pd.DataFrame()
                # get if this is last or not
                end=time.time()
                last = (idx == (numlayers -1))
                df = df.append({
                    "dataset":      args.dataset,
                    "Network":      args.arch,
                    "Mapping_mode": args.mapping_mode,
                    "arraySize":    args.arraySize,
                    "abits":        args.abits,
                    "wbits":        args.wbits,
                    "psum_mode":    args.psum_mode,
                    "pbits":        args.pbits,
                    "pquant_idx":   idx,
                    "last":         last,
                    "center":       center,
                    "best_pbound":  best_pbound,
                    "Test Top1":   top1['test'],
                    "Layer time":   end-layer_time,
                    "Total Time":   end-start_time
                    }, ignore_index=True)
                df.to_pickle(report_file)
                df.to_csv(report_path+'/accuracy_report.txt', sep = '\t', index = False) # last result remain

            # reset DALI iterators
            if args.dali:
                valid_loader.reset()

        print('\nPsum Parameter Search Time | Total: {total_time}s | Layer: {layer_time}s | pbound: {pbound_time}s'.format(
                    total_time=time.time()-start_time,
                    layer_time=time.time()-layer_time,
                    pbound_time=time.time()-pbound_time))
        return
    else:
        print("This file only support evaluation mode!")
        return

if __name__ == '__main__':
    main()
