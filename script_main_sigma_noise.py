# example
# cifar10: python script_main_sigma_noise.py -g 0 --arraySize 128 --iter 5 --trained_noise 0.03 -nt
# imagenet: python script_main_sigma_noise.py -g 0 --dataset imagenet --arch psum_resnet18 --arraySize 128 --pclip_custom --iter 1
import os
import sys
import getopt
import argparse

parser = argparse.ArgumentParser(description = 'run multiple times')

parser.add_argument('-g', '--gpu-id', default=0, type=int)
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['mnist', 'cifar10', 'cifar100', 'imagenet', 'vww'],
                    help='Dataset to be used for training.')
parser.add_argument('--arch', default='psum_vgg9', type=str,
                    choices=['psum_vgg9', 'psum_resnet18', 'psum_alexnet', 'psum_yolov2'])
parser.add_argument('--mapping_mode', default='2T2R', type=str)
parser.add_argument('--arraySize', type=int, nargs='+', default=[128, 256, 512, 1024],
                    help='arraySize list')
parser.add_argument('--pbits', type=float, nargs='+', default=[8, 7, 6, 5, 4, 3, 2],
                    help='pbits list')
parser.add_argument('--pclip_custom', action='store_true',
                    help='Store weight hist')    
parser.add_argument('--pclip', type=str, nargs='+', default=['sigma'],
                    help='pclip list')
parser.add_argument('--trained_noise', type=float, nargs='+', default=[0.0])
parser.add_argument('--inf_noise', type=float, nargs='+', default=[0.01, 0.02, 0.03, 0.04, 0.05])
parser.add_argument('--ratio', type=int, default=100,
                    help='ratio list')
parser.add_argument('--iter', default=5, type=int,
                    help='per_class list')
parser.add_argument('--server', default='val1', type=str)
parser.add_argument('--whist', action='store_true',
                    help='Store weight hist')
parser.add_argument('-nt', '--noise_train', action='store_true',
                    help='Store weight hist')    
parser.add_argument('-rw', '--log_rw', action='store_true',
                    help='rewrite log file')    



args = parser.parse_args()

# parameter setting
argfile = './argfiles/args_{}_noise'.format(args.arch)

if args.dataset == 'cifar10':
    per_class = 500
    if args.arch == 'psum_vgg9':
        check_file = "layer6_hist.pkl"
    else:
        assert False, "Check arch model, This model can't support in this script"
    
    data = None # default
    pclip_list = args.pclip

elif args.dataset == 'imagenet':
    per_class = 50
    if args.arch == 'psum_alexnet':
        check_file = "layer5_hist.pkl"
    elif args.arch == 'psum_resnet18':
        check_file = "layer15_hist.pkl"
    else:
        assert False, "Check arch model, This model can't support in this script"

    if args.server == 'val7':
        data = "/home/data/imagenet_honesty/images/"
    if args.server == 'val5':
        data = "/home/data/imagenet/images/"
    else:
        data = "/home/data/images/"

    if args.pclip_custom:
        if (args.mapping_mode == '2T2R') or (args.mapping_mode == 'ref_a'):
            pclip_list = ['max', 'max', 'max', 'sigma', 'sigma', 'sigma', 'sigma']
        elif (args.mapping_mode == 'PN'):
            pclip_list = ['max', 'max', 'max', 'max', 'sigma', 'sigma', 'sigma']
        elif (args.mapping_mode == 'ref_d'):
            pclip_list = ['sigma', 'sigma', 'sigma', 'sigma', 'sigma', 'sigma', 'sigma']
        elif (args.mapping_mode == 'two_com'):
            pclip_list = ['sigma', 'sigma', 'sigma', 'sigma', 'sigma', 'sigma', 'sigma']
        else:
            assert False
    else:
        pclip_list = args.pclip

else:
    assert False, "The dataset is not supported"

arraySize_list = args.arraySize
pbits_list = args.pbits
noise_param_list = args.inf_noise
ratio = args.ratio
noise_type = 'dynamic'
local = None
iterations = args.iter

for trained_noise in args.trained_noise:
    for arraysize in arraySize_list:
        # setting pretrained
        if args.noise_train:
            pretrained = "./checkpoints/{}/{}/{}/{}_c:4/a:4_w:4/trained_noise_{}_ratio_{}/log_bitserial_info/model_best.pth.tar".format(
                args.dataset, args.arch, args.mapping_mode, arraysize, trained_noise, ratio)
            noise_bool = 'y'
        else:
            if args.dataset == 'cifar10':
                pretrained = "./checkpoints/cifar10/lsq_vgg9/a:4_w:4/2022-Apr-01-18-57-49/model_best.pth.tar"
            elif args.dataset == 'imagenet':
                if args.arch == "psum_alexnet":
                    # weight asymmetric distribution (-8 ~ +7)
                    pretrained= "./checkpoints/imagenet/lsq_alexnet/a:4_w:4/2022-Jun-28-20-54-44/model_best.pth.tar"
                    # weight symmetric distribution (-7 ~ +7)
                    # pretrained = "./checkpoints/imagenet/lsq_alexnet/a:4_w:4/2022-Jun-29-23-47-56/model_best.pth.tar"
                elif args.arch == "psum_resnet18":
                    pretrained = "./checkpoints/imagenet/lsq_resnet18/a:4_w:4/2022-Sep-12-01-10-42/model_best.pth.tar"
                else:
                    assert False, "Check dataset input"
            noise_bool = 'n'

        for noise_param in noise_param_list:

            if args.log_rw:
                log_file=True
                local = "/home/nameunkang/Project/QNN_CIM"
            else:
                if args.noise_train:
                    log_path = os.path.join("checkpoints", args.dataset, "eval", args.arch, args.mapping_mode, "{}_c:4/a:4_w:4/class_split_per_{}/noise_std_{}/noise_{}_ratio_{}_train_{}/log_bitserial_info/hist".format(arraysize, per_class, noise_type, noise_param, ratio, trained_noise), check_file)
                else:
                    log_path = os.path.join("checkpoints", args.dataset, "eval", args.arch, args.mapping_mode, "{}_c:4/a:4_w:4/class_split_per_{}/noise_std_{}/noise_{}_ratio_{}/log_bitserial_info/hist".format(arraysize, per_class, noise_type, noise_param, ratio), check_file)
                
                if os.path.isfile(log_path):
                    log_file=False 
                else:
                    log_file=True
                    local = "/home/nameunkang/Project/QNN_CIM"
            
            for i, pbit in enumerate(pbits_list):
                for iter in range(iterations):
                    num = i if args.pclip_custom else 0
                    print(f'iter {iter+1}, pbits {pbit}, arraySize {arraysize}, pclip {pclip_list[num]}, trained_noise {trained_noise}, noise_param {noise_param}, log_file {log_file}, local {local}')
                    if data:
                        if local:
                            os.system('python main.py --argfile {} --data {} --gpu-id {} --psum_mode sigma --arraySize {} --mapping_mode {} \
                                        --pbits {} --pclip {} --per_class {} --testlog_reset {} --log_file {} --pretrained {} \
                                        -loc {} -n y --noise_type {} --noise_param {} --trained_noise {} --ratio {} --whist {} -nt {}'
                                        .format(argfile, data, args.gpu_id, arraysize, args.mapping_mode, pbit, pclip_list[num], per_class, log_file, log_file, pretrained,
                                                local, noise_type, noise_param, trained_noise, ratio, args.whist, noise_bool))
                            local=None
                        else:
                            os.system('python main.py --argfile {} --data {} --gpu-id {} --psum_mode sigma --arraySize {} --mapping_mode {} \
                                    --pbits {} --pclip {} --per_class {} --testlog_reset {} --log_file {} --pretrained {} -n y --noise_type {} --noise_param {} --trained_noise {} --ratio {} --whist {} -nt {}'
                                    .format(argfile, data, args.gpu_id, arraysize, args.mapping_mode, pbit, pclip_list[num], per_class, log_file, log_file, pretrained, 
                                            noise_type, noise_param, trained_noise, ratio, args.whist, noise_bool))
                    else:
                        if local:
                            os.system('python main.py --argfile {} --gpu-id {} --psum_mode sigma --arraySize {} --mapping_mode {} \
                                        --pbits {} --pclip {} --per_class {} --testlog_reset {} --log_file {} --pretrained {} -loc {} -n y --noise_type {} --noise_param {} --trained_noise {} --ratio {} --whist {} -nt {}'
                                        .format(argfile, args.gpu_id, arraysize, args.mapping_mode, pbit, pclip_list[num], per_class, log_file, log_file, pretrained,
                                                local, noise_type, noise_param, trained_noise, ratio, args.whist, noise_bool))
                            local=None
                        else:
                            os.system('python main.py --argfile {} --gpu-id {} --psum_mode sigma --arraySize {} --mapping_mode {} \
                                    --pbits {} --pclip {} --per_class {} --testlog_reset {} --log_file {} --pretrained {} -n y --noise_param {} --trained_noise {} --ratio {} --whist {} -nt {}'
                                    .format(argfile, args.gpu_id, arraysize, args.mapping_mode, pbit, pclip_list[num], per_class, log_file, log_file, pretrained, noise_param, trained_noise, ratio, args.whist, noise_bool))
                    log_file=False
