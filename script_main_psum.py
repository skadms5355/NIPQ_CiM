# noisy cifar-10 in LSQ model 
# python script_main.py --argfile ./argfiles/args_psum_vgg9_lsq -g 0 --psum_comp --co_noise 1 --noise_type interp --iter 5
# noisy imangeNet 
# python script_main.py --argfile ./argfiles/args_psum_resnet18 --dataset imagenet --psum_comp -g 0 --co_noise 1 --noise_type interp --iter 10
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
parser.add_argument('--testlog', action='store_true')
parser.add_argument('--arraySize', type=int, nargs='+', default=[128],
                    help='arraySize list')
parser.add_argument('--pbits', type=float, nargs='+', default=[2],
                    help='pbits list')
parser.add_argument('--psum_mode', type=str, default='retrain',
                    help='psum mode test')
parser.add_argument('--is_noise', action='store_true')
parser.add_argument('--co_noise', type=float, nargs='+', default=[0])
parser.add_argument('--noise_type', default='interp', type=str,
                    choices=['static', 'grad', 'prop', 'interp', 'hynix_std'])
parser.add_argument('--shrink', type=float, default=None)
parser.add_argument('--deltaG', type=float, default=None)
parser.add_argument('--retention', type=bool, default=False)
parser.add_argument('--reten_type', type=str, default='percent',
                    choices=['percent', 'static', 'invert_p'])
parser.add_argument('--reten_val', type=float, default=0)
parser.add_argument('--iter', default=1, type=int,
                    help='how many iterate inference process')
parser.add_argument('--tnoise', action='store_true')
parser.add_argument('--tco_noise', type=float, default=0)
parser.add_argument('--tdeltaG', type=float, default=None)
parser.add_argument('--is_ADCn', action='store_true')
parser.add_argument('--ADC_std', type=float, nargs='+', default=[0])

args = parser.parse_args()

pbits_list = args.pbits
mapping_mode_list = args.mapping_mode
arraySize = args.arraySize
co_noise_list = args.co_noise
ADC_std_list = args.ADC_std

if "vgg9" in args.argfile:
    model = 'vgg9'
    if "quant" in args.argfile:
        arch = "psum_lsq_vgg9"
        if "pst" in args.argfile:
            model_mode = 'quant_pst'
        else:
            model_mode = 'quant'
    elif "pst" in args.argfile:
        arch = "psum_lsq_vgg9_train"
        if "pnq" in args.argfile:
            model_mode = 'pnq_pst'
        elif "lsq" in args.argfile:
            model_mode = 'lsq_pst'
        else:
            assert False, "Check argfile name"
    else:
        assert False, "Check argfile file name"
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

# psum_operation
for pbit in pbits_list:
    for co_noise in co_noise_list:
        for mapping_mode in mapping_mode_list:
            if args.dataset == 'cifar10':
                per_class = 500
                if model_mode == 'quant': # Baseline
                    if mapping_mode == "2T2R":
                        if args.tnoise:
                            if args.tco_noise == 2:
                                pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/2T2R/no_psum_c:4/hynix_std_rel_2.0/2024-Feb-14-15-59-46/model_best.pth.tar'
                            elif args.tco_noise == 4:
                                pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/2T2R/no_psum_c:4/hynix_std_rel_4.0/2024-Jan-19-18-10-12/model_best.pth.tar'
                                if args.tdeltaG is not None:
                                    if args.tdeltaG == 8:
                                        pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/2T2R/no_psum_c:4/hynix_std_rel_4.0_deltaG_8.0/2024-Feb-19-18-52-29/model_best.pth.tar'
                                    elif args.tdeltaG == 5:
                                        pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/2T2R/no_psum_c:4/hynix_std_rel_4.0_deltaG_5.0/2024-Feb-21-19-48-35/model_best.pth.tar'
                            else:
                                assert False, "No trained model at {} tco_noise".format(args.tco_noise)
                        else:
                            pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/2022-Apr-01-18-57-49/model_best.pth.tar'
                    elif mapping_mode == 'ref_a':
                        if args.tnoise:
                            if args.tco_noise == 2:
                                assert False, "No trained model"
                                # pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/ref_a/no_psum_c:4/hynix_std_rel_2.0/2024-Feb-14-15-59-46/model_best.pth.tar'
                            elif args.tco_noise == 4:
                                pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/ref_a/no_psum_c:4/hynix_std_rel_4.0/2024-Feb-23-16-02-24/model_best.pth.tar'
                                # if args.tdeltaG is not None:
                                #     if args.tdeltaG == 8:
                                #         pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/ref_a/no_psum_c:4/hynix_std_rel_4.0_deltaG_8.0/2024-Feb-19-18-52-29/model_best.pth.tar'
                                #     elif args.tdeltaG == 5:
                                #         pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/ref_a/no_psum_c:4/hynix_std_rel_4.0_deltaG_5.0/2024-Feb-21-19-48-35/model_best.pth.tar'
                            else:
                                assert False, "No pretrained model at {} tco_noise".format(args.tco_noise)
                        else:
                            pretrained = './checkpoints/cifar10/quant/lsq_vgg9/a:4_w:4/2022-Apr-01-18-57-49/model_best.pth.tar'
                    else:
                        assert False, "No trained model at {} mode".format(mapping_mode)
                elif model_mode == 'quant_pst': #partial sum training without group operations 
                    if mapping_mode == '2T2R':
                        if args.tnoise:
                            if args.tco_noise == 2:
                                pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/2T2R/128_c:4/hynix_std_type_2.0/2024-Feb-14-16-44-32/model_best.pth.tar'
                            elif args.tco_noise == 4:
                                pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/2T2R/128_c:4/hynix_std_type_4.0/2024-Jan-04-17-09-19/model_best.pth.tar'
                                if args.tdeltaG is not None:
                                    if args.tdeltaG == 8:
                                        pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/2T2R/128_c:4/hynix_std_type_4.0_deltaG_8.0/2024-Feb-19-17-00-11/model_best.pth.tar'
                                    elif args.tdeltaG == 5:
                                        pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/2T2R/128_c:4/hynix_std_type_4.0_deltaG_5.0/2024-Feb-22-11-13-02/model_best.pth.tar'
                            else:
                                assert False, "No pretrained model at {} tco_noise".format(args.tco_noise)
                        else:
                            pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/2T2R/128_c:4/2023-Nov-10-18-12-17/model_best.pth.tar'
                    elif mapping_mode == 'ref_a':
                        if args.tnoise:
                            if args.tco_noise == 2:
                                assert False, "No trained model"
                                # pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/ref_a/128_c:4/hynix_std_type_2.0/2024-Feb-14-16-44-32/model_best.pth.tar'
                            elif args.tco_noise == 4:
                                pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/ref_a/128_c:4/hynix_std_type_4.0/2024-Feb-23-18-55-29/model_best.pth.tar'
                                # if args.tdeltaG is not None:
                                #     if args.tdeltaG == 8:
                                #         pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/ref_a/128_c:4/hynix_std_type_4.0_deltaG_8.0/2024-Feb-19-17-00-11/model_best.pth.tar'
                                #     elif args.tdeltaG == 5:
                                #         pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/ref_a/128_c:4/hynix_std_type_4.0_deltaG_5.0/2024-Feb-22-11-13-02/model_best.pth.tar'
                            else:
                                assert False, "No trained model at {} tco_noise".format(args.tco_noise)
                        else:
                            pretrained = './checkpoints/cifar10/quant_pst/psum_lsq_vgg9/a:4_w:4/2T2R/128_c:4/2023-Nov-10-18-12-17/model_best.pth.tar'
                    else:
                        assert False, "No trained model at {} mode".format(mapping_mode)
                elif model_mode == 'lsq_pst':
                    if pbit == 2:
                        if mapping_mode == '2T2R':
                            if args.tnoise:
                                if args.tco_noise == 2:
                                    pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/hynix_std_type_2.0/2024-Feb-14-14-35-43/model_best.pth.tar'
                                elif args.tco_noise == 4:
                                    pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/hynix_std_type_4.0/2024-Jan-16-18-06-38/model_best.pth.tar'
                                    if args.tdeltaG is not None:
                                        if args.tdeltaG == 8:
                                            pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/hynix_std_type_4.0_deltaG_8.0/2024-Feb-16-20-48-01/model_best.pth.tar'
                                        elif args.tdeltaG == 5:
                                            pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/hynix_std_type_4.0_deltaG_5.0/2024-Feb-22-13-46-17/model_best.pth.tar'
                                else:
                                    assert False, "No pretrained model at {} tco_noise".format(args.tco_noise)
                            else:
                                pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/2023-Nov-10-18-13-38/model_best.pth.tar'
                        elif mapping_mode == 'ref_a':
                            if args.tnoise:
                                if args.tco_noise == 2:
                                    assert False, "No trained model"
                                    # pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/ref_a/128_c:4/hynix_std_type_2.0/2024-Feb-14-14-35-43/model_best.pth.tar'
                                elif args.tco_noise == 4:
                                    pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/ref_a/128_c:4/hynix_std_type_4.0/2024-Feb-26-17-02-41/model_best.pth.tar'
                                    # if args.tdeltaG is not None:
                                    #     if args.tdeltaG == 8:
                                    #         pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/ref_a/128_c:4/hynix_std_type_4.0_deltaG_8.0/2024-Feb-16-20-48-01/model_best.pth.tar'
                                    #     elif args.tdeltaG == 5:
                                    #         pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/ref_a/128_c:4/hynix_std_type_4.0_deltaG_5.0/2024-Feb-22-13-46-17/model_best.pth.tar'
                                else:
                                    assert False, "No trained model at {} tco_noise".format(args.tco_noise)
                            else:
                                pretrained = './checkpoints/cifar10/lsq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/2023-Nov-10-18-13-38/model_best.pth.tar'
                        else:
                            assert False, "No pretrained model at {} mode".format(mapping_mode)
                    else:
                        assert False, "No pretrained model with {pbit} pbit"
                elif model_mode == 'pnq_pst':
                    if pbit == 2:
                        if mapping_mode == '2T2R':
                            if args.tnoise:
                                if args.tco_noise == 2:
                                    pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/hynix_std_type_2.0/2024-Feb-20-17-04-46/model_best_LSQ.pth.tar'
                                elif args.tco_noise == 4:
                                    pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/hynix_std_type_4.0/2024-Jan-30-10-56-00/model_best.pth.tar'
                                    if args.tdeltaG is not None:
                                        if args.tdeltaG == 8:
                                            pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/hynix_std_type_4.0_deltaG_8.0/2024-Feb-20-18-52-26/model_best_LSQ.pth.tar'
                                        elif args.tdeltaG == 5:
                                            pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/hynix_std_type_4.0_deltaG_5.0/2024-Feb-22-13-27-56/model_best_LSQ.pth.tar'
                            else:
                                pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/2023-Nov-15-19-47-39/model_best.pth.tar'
                                # pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/2024-Feb-15-18-45-36/model_best.pth.tar'
                        elif mapping_mode == 'ref_a':
                            if args.tnoise:
                                if args.tco_noise == 2:
                                    assert False, "No trained model"
                                    # pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/ref_a/128_c:4/hynix_std_type_2.0/2024-Feb-20-17-04-46/model_best_LSQ.pth.tar'
                                elif args.tco_noise == 4:
                                    pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/ref_a/128_c:4/hynix_std_type_4.0/2024-Feb-23-15-27-34/model_best_LSQ.pth.tar'
                                    if args.tdeltaG is not None:
                                        if args.tdeltaG == 8:
                                            pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/ref_a/128_c:4/hynix_std_type_4.0_deltaG_8.0/2024-Feb-20-18-52-26/model_best_LSQ.pth.tar'
                                        elif args.tdeltaG == 5:
                                            pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/ref_a/128_c:4/hynix_std_type_4.0_deltaG_5.0/2024-Feb-22-13-27-56/model_best_LSQ.pth.tar'
                            else:
                                pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/2023-Nov-15-19-47-39/model_best.pth.tar'
                                # pretrained = './checkpoints/cifar10/pnq_pst/psum_lsq_vgg9_train/a:4_w:4/2T2R/128_c:4/2024-Feb-15-18-45-36/model_best.pth.tar'
                        else:
                            assert False, "No pretrained model at {} mode".format(mapping_mode)
                    else:
                        assert False, "No pretrained model with {pbit} pbit"
                else:
                    assert False, "No pretrained model at {model_mode}"

            for a_size in arraySize:
                for ADC_std in ADC_std_list:
                    testlog=args.testlog
                    if "sigma" in args.psum_mode:
                        noise_name = 'type_{}'.format(co_noise)
                        if args.shrink is not None:
                            noise_name = '{}_shrink_{}'.format(noise_name, args.shrink)
                        if args.deltaG is not None:
                            noise_name = '{}_deltaG_{}'.format(noise_name, args.deltaG)
                        if args.is_ADCn:
                            noise_name = '{}_ADC_{}'.format(noise_name, ADC_std)

                        if args.is_noise:
                            if args.tnoise:
                                tn_file = 'tnoise_{}'.format(args.tco_noise)
                                if args.tdeltaG is not None:
                                    tn_file = 'tnoise_{}_G_{}'.format(args.tco_noise, args.tdeltaG)
                                log_path = os.path.join("checkpoints", args.dataset, model_mode, arch, "eval/a:4_w:4", mapping_mode, "{}_c:4/{}_{}_{}/log_bitserial_info/hist".format(a_size, tn_file, args.noise_type, noise_name), check_file)
                            else:
                                log_path = os.path.join("checkpoints", args.dataset, model_mode, arch, "eval/a:4_w:4", mapping_mode, "{}_c:4/{}_{}/log_bitserial_info/hist".format(a_size, args.noise_type, noise_name), check_file)
                        else:
                            log_path = os.path.join("checkpoints", args.dataset, model_mode, arch, "eval/a:4_w:4", mapping_mode, "{}_c:4/log_bitserial_info/hist".format(a_size), check_file)
                        
                        if os.path.isfile(log_path):
                            log_file=False
                        else:
                            log_file=True
                    else:
                        if args.tnoise:
                            tn_file = 'tnoise_{}'.format(args.tco_noise)
                            if args.tdeltaG is not None:
                                tn_file = 'tnoise_{}_G_{}'.format(args.tco_noise, args.tdeltaG)
                        log_file=False

                    if args.is_noise:
                        for i in range(args.iter):
                            print(f'this operation is iter {i+1} co_noise {co_noise} ADC_std {ADC_std} pbit {pbit}, arraySize {a_size}, model_mode {model_mode} log_file {log_file} noise_type {args.noise_type}')
                            if args.shrink is None:
                                if args.deltaG is None:
                                    if args.tnoise:
                                        if not args.is_ADCn:
                                            os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                        --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                        --is_noise y --co_noise {} --noise_type {} --tn_file {} --retention {} --reten_type {} --reten_val {}'
                                                        .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, tn_file, args.retention, args.reten_type, args.reten_val))
                                        else:
                                            os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                        --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                        --is_noise y --co_noise {} --noise_type {} --tn_file {} --retention {} --reten_type {} --reten_val {} \
                                                        --is_ADCn y --ADC_std {}'
                                                        .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, tn_file, args.retention, args.reten_type, args.reten_val, ADC_std))
                                    else:
                                        if not args.is_ADCn:
                                            os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                        --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                        --is_noise y --co_noise {} --noise_type {} --retention {} --reten_type {} --reten_val {}'
                                                        .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, args.retention, args.reten_type, args.reten_val))
                                        else:
                                            os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                        --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                        --is_noise y --co_noise {} --noise_type {} --retention {} --reten_type {} --reten_val {} \
                                                        --is_ADCn y --ADC_std {}'
                                                        .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, args.retention, args.reten_type, args.reten_val, ADC_std))
                                else: 
                                    if args.tnoise:
                                        if not args.is_ADCn:
                                            os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                        --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                        --is_noise y --co_noise {} --noise_type {} --tn_file {} --deltaG {} --retention {} --reten_type {} --reten_val {}'
                                                        .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, tn_file, args.deltaG, args.retention, args.reten_type, args.reten_val))
                                        else:
                                            os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                        --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                        --is_noise y --co_noise {} --noise_type {} --tn_file {} --deltaG {} --retention {} --reten_type {} --reten_val {} \
                                                        --is_ADCn y --ADC_std {}'
                                                        .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, tn_file, args.deltaG, args.retention, args.reten_type, args.reten_val, ADC_std))
                                    else:
                                        if not args.is_ADCn:
                                            os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                        --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                        --is_noise y --co_noise {} --noise_type {} --deltaG {} --retention {} --reten_type {} --reten_val {}'
                                                        .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, args.deltaG, args.retention, args.reten_type, args.reten_val))
                                        else:
                                            os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                        --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                        --is_noise y --co_noise {} --noise_type {} --deltaG {} --retention {} --reten_type {} --reten_val {} \
                                                        --is_ADCn y --ADC_std {}'
                                                        .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, args.deltaG, args.retention, args.reten_type, args.reten_val, ADC_std))
                            else:
                                if args.tnoise:
                                    os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                --is_noise y --co_noise {} --noise_type {} --tn_file {} --shrink {} --retention {} --reten_type {} --reten_val {}'
                                                .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, tn_file, args.shrink, args.retention, args.reten_type, args.reten_val))
                                else:
                                    os.system('python main.py  --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                                --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} \
                                                --is_noise y --co_noise {} --noise_type {} --shrink {} --retention {} --reten_type {} --reten_val {}'
                                                .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained, co_noise, args.noise_type, args.shrink, args.retention, args.reten_type, args.reten_val))
                            testlog=False
                            log_file=False
                    else:
                        print(f'this operation is pbit {pbit}, arraySize {a_size}, per_class {per_class}, testlog_reset {testlog} log_file {log_file}')
                        os.system('python main.py --argfile {} --gpu-id {} --arraySize {} --mapping_mode {} \
                                    --pbits {} --per_class {} --psum_mode {} --testlog_reset {} --log_file {} --pretrained {} --is_noise n --nipq_noise qnoise'
                                    .format(args.argfile, args.gpu_id, a_size, mapping_mode, pbit, per_class, args.psum_mode, testlog, log_file, pretrained))
                        testlog=False
                        log_file=False


