import os
import sys
import getopt
import argparse

parser = argparse.ArgumentParser(description = 'run multiple times')

parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['mnist', 'cifar10', 'cifar100', 'imagenet', 'vww'],
                    help='Dataset to be used for training.')
parser.add_argument('--argfile', default='./argfiles/args_vgg9_nipq_hnoise_inf', type=str)
parser.add_argument('-g', '--gpu-id', default=0, type=int)
parser.add_argument('--mapping_mode', type=str,  nargs='+', default=['2T2R'])
parser.add_argument('--psum_comp', action='store_true')
parser.add_argument('--arraySize', type=int, nargs='+', default=[128, 256, 512, 1024],
                    help='arraySize list')
parser.add_argument('--pbits', type=float, nargs='+', default=[32, 8, 7, 6, 5, 4, 3, 2],
                    help='pbits list')
parser.add_argument('--pclip', type=str, nargs='+', default=['sigma'],
                    help='pclip list')
parser.add_argument('--co_noise', type=float, nargs='+', default=[0.01, 0.02, 0.03, 0.04, 0.05])
parser.add_argument('--noise_type', default='prop', type=str,
                    choices=['static', 'grad', 'prop'])

args = parser.parse_args()

if args.psum_comp:
    pbits_list = args.pbits
    pclip_list = args.pclip
    arraySize = args.arraySize
    local = None

mapping_mode_list = args.mapping_mode
co_noise_list = args.co_noise
noise_type = args.noise_type

if "vgg9" in args.argfile:
    if "nipq" in args.argfile:
        arch = "nipq_vgg9"
    elif "psum" in args.argfile:
        arch = "psum_vgg9"
    check_file = "layer6_hist.pkl"
elif "psum_alexnet" in args.argfile:
    arch = "psum_alexnet"
    check_file = "layer5_hist.pkl"
else:
    arch = None
    check_file = None

if args.dataset == 'imagenet':
    per_class = 50
    if arch == 'psum_alexnet':
        pass
    elif arch == 'psum_resnet18':
        pass
elif args.dataset == 'pascal':
    per_class = 0
    pass
elif args.dataset == 'cifar10':
    pretrained = './checkpoints/cifar10/nipq/nipq_vgg9/qnoise_fix:4/2023-Feb-22-14-31-20/model_best.pth.tar'


if not args.psum_comp:
    for mapping_mode in mapping_mode_list:
        for co_noise in co_noise_list:
            print(f'this operation is co_noise {co_noise} in mapping_mode {mapping_mode}')
            os.system('python main_nipq.py --argfile {} --gpu-id {} --psum_comp {} --mapping_mode {} --pretrained {} \
                    --co_noise {} --noise_type {}'
                    .format(args.argfile, args.gpu_id, args.psum_comp, mapping_mode, pretrained, co_noise, noise_type))

# psum_operation
else:
    for a_size in arraySize:
        testlog=True
        if args.mapping_mode == "2T2R":
            log_path = os.path.join("checkpoints", args.dataset, "eval", arch, args.mapping_mode, "{}_c:4/a:4_w:4/class_split_per_{}/{}/log_bitserial_info/hist".format(a_size, per_class, weight_sym), check_file)
        else:
            log_path = os.path.join("checkpoints", args.dataset, "eval", arch, args.mapping_mode, "{}_c:4/a:4_w:4/class_split_per_{}/log_bitserial_info/hist".format(a_size, per_class), check_file)
        

        if os.path.isfile(log_path):
            log_file=False
        else:
            log_file=True
            local = "/home/nameunkang/Project/QNN_CIM"

        for pclip in pclip_list:
            for pbit in pbits_list:
                print(f'this operation is pbits {pbit}, arraySize {a_size}, per_class {per_class}, testlog_reset {testlog} log_file {log_file}')
                if local:
                    os.system('python main.py  --argfile {} --gpu-id {} --psum_mode sigma --arraySize {} --mapping_mode {} \
                                --pbits {} --pclip {} --per_class {} --testlog_reset {} --log_file {} --wsymmetric {} --pretrained {} -loc {}'
                                .format(args.argfile, args.gpu_id, a_size, args.mapping_mode, pbit, pclip, per_class, testlog, log_file, args.wsymmetric, pretrained,
                                        local))
                    local=None
                else:
                    os.system('python main.py  --argfile {} --gpu-id {} --psum_mode sigma --arraySize {} --mapping_mode {} \
                            --pbits {} --pclip {} --per_class {} --testlog_reset {} --log_file {} --wsymmetric {} --pretrained {}'
                            .format(args.argfile, args.gpu_id, a_size, args.mapping_mode, pbit, pclip, per_class, testlog, log_file, args.wsymmetric, pretrained))
                testlog=False
                log_file=False


