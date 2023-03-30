"""This page returns network info"""

def get_num_split_layers(arch):
    """ Return number of layers in the CIM arrays"""
    numlayers = 0
    if 'vgg9' in arch:
        numlayers = 7
    else:
        assert False, "No information about #split_layers of {arch}"
    return numlayers

def get_baseline_acc(arch, wbits, abits):
    """ Return baseline test acc """
    baseline_acc = 0
    if arch == 'psum_vgg9':
        if wbits==4 and abits==4:
            baseline_acc = 93.16
        else:
            assert False, f"No information about baseline_acc of {arch}, w:{wbits}/a:{abits}"
    else:
        assert False, f"No information about baseline_acc of {arch}, w:{wbits}/a:{abits}"
    return baseline_acc
