# noisy cifar-10 in LSQ model 
# python script_main.py --argfile ./argfiles/args_psum_vgg9_lsq -g 0 --psum_comp --co_noise 1 --noise_type interp --iter 5

import os
import sys
import getopt
import argparse

parser = argparse.ArgumentParser(description = 'run multiple times')

parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['mnist', 'cifar10', 'cifar100', 'imagenet', 'vww'],
                    help='Dataset to be used for training.')
parser.add_argument('--argfile', default='./argfiles/args_psum_vgg9_nipq_inf', type=str)
parser.add_argument('-g', '--gpu-id', default=0, type=int)
parser.add_argument('--mapping_mode', type=str,  nargs='+', default=['2T2R'])
parser.add_argument('--psum_comp', action='store_true')
parser.add_argument('--testlog', action='store_true')
parser.add_argument('--arraySize', type=int, nargs='+', default=[128],
                    help='arraySize list')
parser.add_argument('--pbits', type=float, nargs='+', default=[32, 8, 7, 6, 5, 4],
                    help='pbits list')
parser.add_argument('--pclip', type=str, nargs='+', default=['sigma'],
                    help='pclip list')
parser.add_argument('--co_noise', type=float, nargs='+', default=[0, 0.03, 0.05])
parser.add_argument('--noise_type', default='prop', type=str,
                    choices=['static', 'grad', 'prop', 'interp'])
parser.add_argument('--iter', default=1, type=int,
                    help='how many iterate inference process')
parser.add_argument('--KD', action='store_true')
parser.add_argument('--FL_quant', action='store_true')
parser.add_argument('--tnipq', default='hwnoise', type=str,
                    choices=['quant', 'hwnoise', 'qhwnoise'])
parser.add_argument('--tnoise_type', default='prop', type=str,
                    choices=['static', 'grad', 'prop', 'interp'])
parser.add_argument('--tres_val', default='abs', type=str,
                    choices=['abs', 'rel'])
parser.add_argument('--tco_noise', type=float, default=0)

args = parser.parse_args()

if args.psum_comp:
    pbits_list = args.pbits
    pclip_list = args.pclip
    arraySize = args.arraySize
    local = None

mapping_mode_list = args.mapping_mode
co_noise_list = args.co_noise
noise_type = args.noise_type
trained_noise = False

if "vgg9" in args.argfile:
    model = 'vgg9'
    if "nipq" in args.argfile:
        arch = "nipq_vgg9"
        if "psum" in args.argfile:
            arch = "psum_nipq_vgg9"
    elif "lsq" in args.argfile:
        arch = "lsq_vgg9"
        if "psum" in args.argfile:
            arch = "psum_lsq_vgg9"
    check_file = "layer6_hist.pkl"
elif "psum_alexnet" in args.argfile:
    model = 'alexnet'
    arch = "psum_alexnet"
    check_file = "layer5_hist.pkl"
elif "resnet18" in args.argfile:
    model = 'resnet18'
    if "nipq" in args.argfile:
        arch = "nipq_resnet18"
        if "psum" in args.argfile:
            arch = "psum_resnet18_nipq"
    elif "lsq" in args.argfile:
        arch = "lsq_resnet18"
        if "psum" in args.argfile:
            arch = "psum_resnet18_lsq"
    check_file = "layer15_hist.pkl"
else:
    arch = None
    check_file = None

if args.dataset == 'imagenet':
    per_class = 50
    if "nipq" in arch:
        pretrained =  './checkpoints/imagenet/nipq/nipq_resnet18/qnoise_fix:4/2023-Apr-03-14-51-48/model_best.pth.tar'
    elif "lsq" in arch:
        if args.FL_quant:
            pretrained  = './checkpoints/imagenet/quant/lsq_resnet18/FL_a:4_w:4/2023-Jun-19-19-18-16/model_best.pth.tar'
        else:
            pretrained  = './checkpoints/imagenet/quant/lsq_resnet18/a:4_w:4/2022-Sep-12-01-10-42/model_best.pth.tar'
    else:
        assert False, "No pretrained model"
elif args.dataset == 'pascal':
    per_class = 0
elif args.dataset == 'cifar10':
    per_class = 500
    if "nipq" in arch:
        pretrained = './checkpoints/cifar10/nipq/nipq_vgg9/qnoise_fix:4/2023-Feb-22-14-31-20/model_best.pth.tar'
    else:
        if args.FL_quant:
            pretrained = './checkpoints/cifar10/quant/lsq_vgg9/FL_a:4_w:4/2023-Jun-19-16-49-05/model_best.pth.tar'
        else:
            pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/2022-Apr-01-18-57-49/model_best.pth.tar'

if not args.psum_comp:
    for mapping_mode in mapping_mode_list:
        for co_noise in co_noise_list:
            print(f'this operation is co_noise {co_noise} in mapping_mode {mapping_mode}')
            os.system('python main_nipq.py --argfile {} --gpu-id {} --psum_comp {} --mapping_mode {} --pretrained {} \
                    --co_noise {} --noise_type {}'
                    .format(args.argfile, args.gpu_id, args.psum_comp, mapping_mode, pretrained, co_noise, noise_type))

# psum_operation
else:
    for mapping_mode in mapping_mode_list:
        if args.tnipq=="qhwnoise":
            if "nipq" in arch:
                tn_file = 't_{}_{}'.format(args.tnoise_type, args.tco_noise)
                pretrained = './checkpoints/{}/nipq/nipq_{}/qhwnoise_fix:4/{}/no_psum_c:4/{}_{}/best_model/model_best.pth.tar'.format(args.dataset, model, mapping_mode, args.tnoise_type, args.tco_noise)
            elif "lsq" in arch:
                if args.KD:
                    tn_file = 'KD_{}_{}_{}'.format(args.tnoise_type, args.tres_val, args.tco_noise)
                    pretrained = './checkpoints/{}/quant/lsq_{}/a:4_w:4/{}/no_psum_c:4/{}_{}_{}/KD_best_model/model_best.pth.tar'.format(args.dataset, model, mapping_mode, args.tnoise_type, args.tres_val, args.tco_noise)
                else:
                    tn_file = '{}_{}_{}'.format(args.tnoise_type, args.tres_val, args.tco_noise)
                    pretrained = './checkpoints/{}/quant/lsq_{}/a:4_w:4/{}/no_psum_c:4/{}_{}_{}/best_model/model_best.pth.tar'.format(args.dataset, model, mapping_mode, args.tnoise_type, args.tres_val, args.tco_noise)
        else:
            tn_file = None

        for co_noise in co_noise_list:
            if co_noise == 0:
                is_noise = False
                nipq_noise = "qnoise"
            else:
                is_noise = True
                nipq_noise = "hwnoise"
                if args.tnipq == "qhwnoise":
                    nipq_noise = "qhwnoise"

            for a_size in arraySize:
                testlog=args.testlog
                if tn_file is None:
                    noise_name = noise_type
                else:
                    noise_name = '{}_{}'.format(tn_file, noise_type)
                if "nipq" in args.argfile:
                    if is_noise:
                        log_path = os.path.join("checkpoints", args.dataset, "nipq", arch, "eval", "{}_fix:4".format(nipq_noise), mapping_mode, "{}_c:4/{}_type_{}/log_bitserial_info/hist".format(a_size, noise_name, co_noise), check_file)
                    else:
                        log_path = os.path.join("checkpoints", args.dataset, "nipq", arch, "eval", "{}_fix:4".format(nipq_noise), mapping_mode, "{}_c:4/log_bitserial_info/hist".format(a_size), check_file)
                else:
                    if args.FL_quant:
                        if is_noise:
                            log_path = os.path.join("checkpoints", args.dataset, "quant", arch, "eval/FL_a:4_w:4", mapping_mode, "{}_c:4/{}_type_{}/log_bitserial_info/hist".format(a_size, noise_name, co_noise), check_file)
                        else:
                            log_path = os.path.join("checkpoints", args.dataset, "quant", arch, "eval/FL_a:4_w:4", mapping_mode, "{}_c:4/log_bitserial_info/hist".format(a_size), check_file)
                    else:
                        if is_noise:
                            log_path = os.path.join("checkpoints", args.dataset, "quant", arch, "eval/a:4_w:4", mapping_mode, "{}_c:4/{}_type_{}/log_bitserial_info/hist".format(a_size, noise_name, co_noise), check_file)
                        else:
                            log_path = os.path.join("checkpoints", args.dataset, "quant", arch, "eval/a:4_w:4", mapping_mode, "{}_c:4/log_bitserial_info/hist".format(a_size), check_file)

                # import pdb; pdb.set_trace()
                if os.path.isfile(log_path):
                    log_file=False
                else:
                    log_file=True
                    # local = "/home/nameunkang/Project/QNN_CIM"

                for pbit in pbits_list:
                    for i in range(args.iter):
                        if is_noise:
                            print(f'this operation is iter {i+1} pbits {pbit}, arraySize {a_size}, per_class {per_class}, testlog_reset {testlog} log_file {log_file} co_noise {co_noise}')
                            if nipq_noise == "qhwnoise":
                                os.system('python main.py  --argfile {} --gpu-id {} --psum_comp {} --arraySize {} --mapping_mode {} \
                                            --pbits {} --per_class {} --testlog_reset {} --log_file {} --pretrained {} \
                                            --is_noise y --tn_file {} --nipq_noise {} --co_noise {} --noise_type {}'
                                            .format(args.argfile, args.gpu_id, args.psum_comp, a_size, mapping_mode, pbit, per_class, testlog, log_file, pretrained, tn_file, nipq_noise, co_noise, noise_type))
                            else:
                                if args.FL_quant:
                                    os.system('python main.py  --argfile {} --gpu-id {} --psum_comp {} --arraySize {} --mapping_mode {} \
                                                --pbits {} --per_class {} --testlog_reset {} --log_file {} --pretrained {} \
                                                --is_noise y --nipq_noise {} --co_noise {} --noise_type {} --FL_quant y'
                                                .format(args.argfile, args.gpu_id, args.psum_comp, a_size, mapping_mode, pbit, per_class, testlog, log_file, pretrained, nipq_noise, co_noise, noise_type))
                                else:
                                    os.system('python main.py  --argfile {} --gpu-id {} --psum_comp {} --arraySize {} --mapping_mode {} \
                                                --pbits {} --per_class {} --testlog_reset {} --log_file {} --pretrained {} \
                                                --is_noise y --nipq_noise {} --co_noise {} --noise_type {}'
                                                .format(args.argfile, args.gpu_id, args.psum_comp, a_size, mapping_mode, pbit, per_class, testlog, log_file, pretrained, nipq_noise, co_noise, noise_type))

                        else:
                            if args.FL_quant:
                                print(f'this operation is pbits {pbit}, arraySize {a_size}, per_class {per_class}, testlog_reset {testlog} log_file {log_file}')
                                os.system('python main.py  --argfile {} --gpu-id {} --psum_comp {} --arraySize {} --mapping_mode {} \
                                            --pbits {} --per_class {} --testlog_reset {} --log_file {} --pretrained {} --is_noise n --nipq_noise {} --FL_quant y'
                                            .format(args.argfile, args.gpu_id, args.psum_comp, a_size, mapping_mode, pbit, per_class, testlog, log_file, pretrained, nipq_noise))
                            else:
                                print(f'this operation is pbits {pbit}, arraySize {a_size}, per_class {per_class}, testlog_reset {testlog} log_file {log_file}')
                                os.system('python main.py  --argfile {} --gpu-id {} --psum_comp {} --arraySize {} --mapping_mode {} \
                                            --pbits {} --per_class {} --testlog_reset {} --log_file {} --pretrained {} --is_noise n --nipq_noise {}'
                                            .format(args.argfile, args.gpu_id, args.psum_comp, a_size, mapping_mode, pbit, per_class, testlog, log_file, pretrained, nipq_noise))
                        testlog=False
                        log_file=False


