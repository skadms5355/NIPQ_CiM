import os
import argparse
import models
import warnings

from utils.generate_valid_from_train_in_imagenet import generate_valid_from_train


__all__ = ['set_arguments', 'check_arguments']

class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            lines = f.read().splitlines()
            arglines = [i.split() for i in lines if not i.startswith("#")]
            argsegs = [seg for sublist in arglines for seg in sublist]
            parser.parse_args(argsegs, namespace)

# str2bool from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_arguments():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(
        description='PyTorch Image Classification Training')

    # Defines the argument groups
    dataset_group = parser.add_argument_group('Dataset options')
    dataloader_group = parser.add_argument_group('Dataloader options')
    distributed_group = parser.add_argument_group('Distributed training options')
    apex_group = parser.add_argument_group('Apex options')
    train_group = parser.add_argument_group('Training options')
    optim_group = parser.add_argument_group('Optimizer options')
    scheduler_group = parser.add_argument_group('LR scheduler options')
    arch_group = parser.add_argument_group('Architecture options')
    psum_group = parser.add_argument_group('Partial sum options')
    noise_group = parser.add_argument_group('Noise options')
    transfer_group = parser.add_argument_group('Transfer Learning options')
    ckpt_group = parser.add_argument_group('Checkpoint options')

    # Datasets
    dataset_group.add_argument('--dataset', default='cifar10', type=str,
                               choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                               help='Dataset to be used for training.')
    dataset_group.add_argument('-d', '--data', default=None, type=str, help='path to dataset')
    dataset_group.add_argument('--valid-size', default=0.1, type=float,
                               help='Ratio of valid set split from training set of CIFAR \
                               dataset. If 0, no validation set used.')
    dataset_group.add_argument('--filelist', default=None, type=str,
                               help='imagenet-path to filelist of valid dataset extracted from train (image_val_from_train_list.txt)')
    dataset_group.add_argument('--mixup', default='False', type=str2bool,
                               help='mix-up pixels of dataset. Used in RtB for cifar100. \
                               If True, using mix-up.')
    dataset_group.add_argument('--alpha', default=1., type=float,
                               help='mixup interpolation coefficient (default: 1)')
    dataset_group.add_argument('--augment', default='True', type=str2bool,
                               help='Data augmentation option. If False, no augmentation')
    dataset_group.add_argument('--zca', default='False', type=str2bool,
                               help='zca whitening augmentation for BNN paper. \
                               If True, using zca whitening.')

    # Data loader options
    dataloader_group.add_argument('-s', '--manualSeed', type=int, help='Manual seed')
    dataloader_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                                  help='Number of data loading workers (default: 4)')

    # DALI options
    dataloader_group.add_argument('--dali', default='False', type=str2bool,
                                  help='Uses DALI dataloader. If True, using DALI.')
    dataloader_group.add_argument('--dali-cpu', default='False', type=str2bool,
                                  help='Runs CPU based version of DALI pipeline.')
    dataloader_group.add_argument('--local_rank', default=0, type=int)

    # distributed training options
    distributed_group.add_argument('--gpu-id', default='0', type=str,
                                   help='ID(s) for CUDA_VISIBLE_DEVICES')
    distributed_group.add_argument('--distributed', default='False', type=str2bool,
                                   help='Use distributed dataparallel')

    # Amp options
    apex_group.add_argument('--amp', default='False', type=str2bool,
                            help='Use automatic mixed precision training.')

    # Training options
    train_group.add_argument('-tm', '--model_mode', default='baseline', type=str,
                             choices=['baseline', 'quant', 'hn_quant', 'nipq'],
                             help='training mode to be used. (default: baseline (basic))')
    train_group.add_argument('--epochs', default=90, type=int, metavar='N',
                             help='Number of total epochs to run')
    train_group.add_argument('--ft_epoch', default=0, type=int, metavar='N',
                             help='quantization tuning epoch in nipq mode')
    train_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                             help='Manual epoch number (useful on restarts)')
    train_group.add_argument('--train-batch', default=256, type=int, metavar='N',
                             help='Train batchsize (default: 256)')
    train_group.add_argument('--test-batch', default=200, type=int, metavar='N',
                             help='Test batchsize (default: 200)')

    # Optimization options
    optim_group.add_argument('-o', '--optimizer', default='sgd', type=str,
                             choices=['sgd', 'adam', 'adamax', 'adamw'],
                             help='Optimizer to be used. (default: sgd)')
    optim_group.add_argument('--drop', '--dropout', default=0, type=float,
                             metavar='Dropout', help='Dropout ratio')
    optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                             help='Momentum')
    optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                             metavar='W', help='Weight decay (default: 1e-4)')
    optim_group.add_argument('--init-method', default='xavier', type=str,
                             choices=['xavier', 'kaiming', 'uniform'],
                             help='which method to use for weight initialization. (default: xavier)')
    optim_group.add_argument('--init-dist', default='normal', type=str,
                             help='which distribution to follow for weight initialization. (default: normal)')
    optim_group.add_argument('--init-fan', default='fan_both', type=str,
                             choices=['fan_in', 'fan_out', 'fan_both'],
                             help='which direction of propagation to optimize. (default: fan_both)')

    # LR scheduler options
    scheduler_group.add_argument('--lr-method', default='lr_step', type=str,
                                 choices=['lr_step', 'lr_linear', 'lr_exp', 'lr_cosineanneal'],
                                 help='Set learning rate scheduling method.')
    scheduler_group.add_argument('--schedule', nargs='+', default=[150, 225], type=int,
                                 help='Decrease learning rate at these epochs when using step method')
    scheduler_group.add_argument('--gamma', default=0.1, type=float,
                                 help='LR is multiplication factor')
    scheduler_group.add_argument('--T0', default=10, type=int,
                                 help='Number of steps for the first restart in SGDR')
    scheduler_group.add_argument('--T-mult', default=1, type=int,
                                 help='A factor increases T_{i} after restart in SGDR')
    scheduler_group.add_argument('--eta_min', default=0.0, type=float,
                                 help='minimum lr of cosine annealing')                            
    scheduler_group.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                                 metavar='LR', help='Initial learning rate')
    scheduler_group.add_argument('--glr', default='False', type=str2bool,
                                 help='glr learning rate scaling for bin conv & linear')
    scheduler_group.add_argument('--warmup', default=0, type=int,
                                 help='Epoch number for warmup scheduler')
    scheduler_group.add_argument('--warmup_start_multiplier', default=0.0, type=float,
                                 help='Epoch number for warmup scheduler')
    scheduler_group.add_argument('--multiplier', default=1.0, type=float,
                                 help='(multiplier) * (base lr) = (target lr) after warmup epochs')

    # Architecture
    arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                            choices=model_names,
                            help='Model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet20)')
    arch_group.add_argument('--input-size', default=224, type=int,
                            help='Input image size')
    arch_group.add_argument('--binary-mode', default='signed', type=str,
                            help='Binary activation mode. "unsigned" for 0/1 activation and "signed" for -1/+1 activation.')
    arch_group.add_argument('--ste', default='clippedrelu', type=str,
                            help='Which ste to use for binary activation.')
    arch_group.add_argument('--init-nonlinear', default=[0.0], type=float, nargs='+',
                            help='Initial values for the nonlinear function trainable parameters')
    arch_group.add_argument('--train-nonlinear', default='False', type=str2bool, 
                            help='Set requires_grad to True for the nonlinear function trainable parameters')
    arch_group.add_argument('--bn-bias', default=0.0, type=float,
                            help='Initialize value for BN bias.')
    arch_group.add_argument('--abits', default=32, type=int,
                            help='Bit resolution of inputs. 32 is for full-precision.')
    arch_group.add_argument('--wbits', default=32, type=int,
                            help='Bit resolution of weights. 32 is for full-precision.')
    arch_group.add_argument('--wsymmetric', default='False', type=str2bool,
                            help='Decide weight symmetric range (Default:False')
    arch_group.add_argument('--x_offset', default=0.0, type=float,
                            help='x-axis offset for activation function.')
    arch_group.add_argument('--width', default=1.0, type=float,
                            help='width for gradient clipping on activation function.')
    arch_group.add_argument('--weight-clip', default=1, type=int,
                            help='0 for no weight clipping. 1 for -1<w<1 clipping.')
    arch_group.add_argument('--weight-scale', default='True', type=str2bool,
                            help='Weight scaling option. False to not use weight scale factor')
    arch_group.add_argument('--padding_mode', default='zeros', type=str,
                            help='zeros: zero-padding, ones: one-padding, alter: alternately padding with +1/-1. \
                            If None, one-padding for signed binary mode and zero-padding for others.')                   
    arch_group.add_argument('--fixed_bit', default=-1, type=int,
                            help='When fix_bit is none mix-precision is trained (default: None) in NIPQ mode')
    #Psum 
    psum_group.add_argument('-p', '--psum_comp', default='False', type=str2bool,
                        help='Psum computation model')
    psum_group.add_argument('--psum_mode', default='sigma', type=str,
                             choices=['sigma', 'scan'],
                             help='psum computation mode')
    psum_group.add_argument('--arraySize', type=int, default=0,
                             metavar='arraySize', help='row count of array for in-memory computing')
    psum_group.add_argument('--mapping_mode', default='none', type=str,
                             choices=['none', '2T2R', 'two_com', 'ref_d', 'ref_a', 'PN'],
                             help='which mapping mode to store weight. (default: none(no special))')
    psum_group.add_argument('--cbits', type=int, default=4,
                            help='Cell bit resolution of arrays. 4 is default.')                        
    psum_group.add_argument('--wbit-serial', default='False', type=str2bool, 
                            help='bit serial computation of weight')
    psum_group.add_argument('--abit-serial', default='False', type=str2bool, 
                            help='bit serial computation of activation') 
    psum_group.add_argument('--log_file', default='False', type=str2bool, 
                            help='Save psum log information')
    psum_group.add_argument('--testlog_reset', default='False', type=str2bool, 
                            help='reset accuracy test log for script')
    psum_group.add_argument('--class-split', default='False', type=str2bool, 
                            help='Class split of training dataset')
    psum_group.add_argument('--pbits', type=float, default=32,
                            help='Bit resolution of partial sums. 32 is for full-precision.')
    psum_group.add_argument('--pclipmode', default='Layer', type=str,
                             choices=['Layer', 'Network'],
                             help='Layer-wise or Network-wise for psum quantization (default: layer-wise)')
    psum_group.add_argument('--pclip', default='sigma', type=str,
                             choices=['sigma', 'max'],
                             help='Clipping range of psum quantization (default: sigma)')
    psum_group.add_argument('--psigma', type=int, default=3,
                            help='Sigma point of clipping range')                         
    psum_group.add_argument('--per_class', default=50, type=int,
                            help='How many watch image(batch) for searching psum distribution mean, std')

    # noise
    noise_group.add_argument('-n', '--is_noise', default='False', type=str2bool,
                            help='Noise effect consideration')
    noise_group.add_argument('--tn_file', default=None, type=str,
                            help='file name of trained weight')
    noise_group.add_argument('--nipq_noise', default='qnoise', type=str, 
                            choices=['qnoise', 'hwnoise', 'qhwnoise'],
                            help='Type of injection noise (default: quant noise) \
                                qnoise: quantization noise (training-inf), hwnoise: quantizatio noise training - hnoise inf \
                                qhwnoise: hnoise + quantizatio noise (training) - hnoise(inf)')                       
    noise_group.add_argument('--co_noise', type=float, default=0.01,
                            help='coefficient of cell noise variation (range: 0.01 ~ 0.05) during inference.')
    noise_group.add_argument('--ratio', type=int, default=100,
                            help='Ratio of Gmax/Gmin (default: 100)')   
    noise_group.add_argument('--noise_type', default='prop', type=str, 
                            choices=['static', 'grad', 'prop', 'meas'],
                            help='Std type of conductance noise (default: static std)')   
    noise_group.add_argument('--meas_type', default='reram', type=str, 
                            choices=['reram', 'mram', 'sram'],
                            help='Type of measured data (default: reram data)')   
    noise_group.add_argument('-nt', '--noise_train', default='False', type=str2bool, 
                            help='Use noise trained weight values')    
    noise_group.add_argument('--res_val', type=str, default='rel',
                            choices=['rel', 'abs'],
                            help='representation methods of resistance (default: relative(rel))).')               

    # Transfer Learning
    transfer_group.add_argument('--transfer-mode', default=0, type=int, choices=[0,1,2,3],
                                help='0: nothing, 1: knowledge-distilation (KD), 2: attention transfer (AT), 3: KD+AT')
    transfer_group.add_argument('--teacher', default=None, type=str, metavar='PATH',
                                help='Path to pretrained teacher model.')
    transfer_group.add_argument('--kd-alpha', default=0.99, type=float,
                                help='Hyperparameter for KD. Weight factor of kd-loss.')
    transfer_group.add_argument('--kd-temperature', default=1.0, type=float,
                                help='Hyperparameter for KD.')
    transfer_group.add_argument('--at-beta', default=1e+3, type=float,
                                help='Hyperparameter for AT. Weight factor of at-loss.')
    
    # Checkpoints
    ckpt_group.add_argument('-c', '--checkpoint', default=None, type=str, metavar='PATH',
                            help='Path to save checkpoint (default: current datetime)')
    ckpt_group.add_argument('-l', '--link', default=None, type=str, metavar='PATH',
                            help='Path to link the default checkpoint to')
    ckpt_group.add_argument('-loc', '--local', default=None, type=str, metavar='PATH', 
                            help='Path to save the default checkpoint to local position')                        
    ckpt_group.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='Path to latest checkpoint (default: none)')
    ckpt_group.add_argument('--report', default='', type=str, metavar='PATH',
                            help='Path to report file (default: none)')
    ckpt_group.add_argument('--whist', default='False', type=str2bool,
                        help='Store weight hist')

    # Miscs
    parser.add_argument('-e', '--evaluate', default='False', type=str2bool,
                        help='Evaluate model on validation set')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='Path to pre-trained model (default: none')
    parser.add_argument('--interpolation', default=None, type=str, 
                        help='interpolation method for data loader. if None, BICUBIC is used.')
    parser.add_argument('--tensorboard', default=0, type=int,
                        choices=[0, 1, 2, 3],
                        help='Use tensorboard with log level specified. (default: 0) \
                        If 0, tensorboard is not used.')

    parser.add_argument('--argfile', type=open, action=LoadFromFile)
    args = parser.parse_args()


    # Use CUDA. This line has to be come before importing torch modules.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    return args

def check_arguments(args):
    # dataset path
    if args.data is None:
        if args.dataset == 'mnist':
            args.data = './data/mnist/'
        elif args.dataset == 'cifar10':
            args.data = './data/cifar10/'
        elif args.dataset == 'cifar100':
            args.data = './data/cifar100/'
        elif args.dataset == 'imagenet':
            args.data = '/home/data/images/'
        elif args.dataset == 'vww':
            args.data = '/home/data/COCO14/'

    # number of classes
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'imagenet':
        args.num_classes = 1000
        if args.psum_comp:
            if args.filelist is None:
                args.filelist = '/mnt/nfs/nameunkang/Project/QNN_CIM/data/imagenet/imagenet_val_from_train_list_per_{}.txt'.format(args.per_class)
            if os.path.isfile(args.filelist):
                print('Exist a valid dataset file from train set')
            else:
                generate_valid_from_train(args)
                print('Generate valid dataset from train dataset')
    elif args.dataset == 'vww':
        args.num_classes = 2

    # dali assertion check
    if args.dali_cpu:
        assert args.dali, 'Please use --dali --dali-cpu together'

    # use amp with DDP only.
    # if args.amp:
    #     assert args.distributed, 'Please use amp only with distributed mode'

    # using distributed mode with CIFAR or MNIST can be slow.
    if args.distributed:
        assert args.dataset == 'imagenet', 'Distributed mode is not supported for dataset other than ImageNet'

    # validation set check
    if args.dataset == 'imagenet' or args.dataset == 'vww':
        if args.valid_size != 0:
            print('\nWe do not support validation split for ImageNet and VisualWakeWords dataset.')
            print('Using valid_size = 0\n')
            args.valid_size = 0
    # label smoothing check
    # if args.label_smoothing:
    #     assert args.loss == 'cross_entropy', 'Currently label smoothing is only supported for Cross Entropy loss function'

    # padding check
    if (args.abits == 1) and (args.binary_mode == 'signed') and (args.padding_mode == 'zeros'):
        warnings.warn("0-padding used while input activation has +1 or -1 values.")
    if (args.abits == 32) and (args.padding_mode != 'zeros'):
        warnings.warn(f"{args.padding_mode}-mode is used instead of 0-padding though activation is in full-precision.")

    # if using lr_exp and gamma is a bit too small, warn
    if args.lr_method == 'lr_exp' and args.gamma < 0.5:
        warnings.warn(f"Using lr_exp with gamma {args.gamma:.3f}. Usually gamma is larger for lr_exp.")

    ## psum mode turn on / off
    if args.psum_comp == False:
        args.wbit_serial = False
        args.abit_serial = False

    if args.evaluate:
        args.teacher = None

    return args
