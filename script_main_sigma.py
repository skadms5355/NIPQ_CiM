import os
import sys
import getopt
import argparse

parser = argparse.ArgumentParser(description = 'run multiple times')

parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['mnist', 'cifar10', 'cifar100', 'imagenet', 'vww'],
                    help='Dataset to be used for training.')
parser.add_argument('--argfile', default='./argfiles/args_psum_vgg9_multi', type=str)
parser.add_argument('-g', '--gpu-id', default=0, type=int)
parser.add_argument('--arraySize', type=int, nargs='+', default=[128, 256, 512, 1024],
                    help='arraySize list')
parser.add_argument('--pbits', type=float, nargs='+', default=[32, 8, 7, 6, 5, 4, 3, 2],
                    help='pbits list')
parser.add_argument('--pclip', type=str, nargs='+', default=['sigma'],
                    help='pclip list')
parser.add_argument('--mapping_mode', default='2T2R', type=str)
parser.add_argument('--wsymmetric', action='store_true',
                    help='Stor weight symmectric') 

args = parser.parse_args()

pbits_list = args.pbits
pclip_list = args.pclip
arraySize = args.arraySize
local = None

if args.wsymmetric:
    weight_sym = 'weight_sym'
else:
    weight_sym = 'weight_asym'

if "psum_vgg9" in args.argfile:
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
        if args.wsymmetric:
            pretrained = "./checkpoints/imagenet/lsq_alexnet/a:4_w:4/2022-Jun-29-23-47-56/model_best.pth.tar"
        else:
            pretrained= "./checkpoints/imagenet/lsq_alexnet/a:4_w:4/2022-Jun-28-20-54-44/model_best.pth.tar"
    elif arch == 'psum_resnet18':
        pretrained= "./checkpoints/imagenet/lsq_resnet18/a:4_w:4/2022-Sep-12-01-10-42/model_best.pth.tar"
elif args.dataset == 'pascal':
    per_class = 0
    pretrained  = "./checkpoints/pascal/quant_yolov2/a:4_w:4/2022-Aug-25-01-59-49/model_best.pth.tar"
elif args.dataset == 'cifar10':
    per_class = 500
    pretrained = "./checkpoints/cifar10/lsq_vgg9/a:4_w:4/2022-Apr-01-18-57-49/model_best.pth.tar"

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


