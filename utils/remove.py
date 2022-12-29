from __future__ import absolute_import
import os
import shutil

__all__ = ['rm_ckpt']

def rm_ckpt(dirpath):
    if os.path.isdir(dirpath):
        checkpoints = os.listdir(dirpath)
        for checkpoint in checkpoints:
            full_checkpoint = os.path.join(dirpath, checkpoint)
            if os.path.isdir(full_checkpoint):
                ckpt = os.path.join(full_checkpoint, 'checkpoint.pth.tar')
                if os.path.isfile(ckpt) == False:
                    shutil.rmtree(full_checkpoint, ignore_errors=True)

def rm_ckpt_all(dirpath):
    checkpoints = os.listdir(dirpath)
    for checkpoint in checkpoints:
        full_checkpoint = os.path.join(dirpath, checkpoint)
        if os.path.isdir(full_checkpoint):
            if checkpoint.startswith('2'):
                ckpt = os.path.join(full_checkpoint, 'checkpoint.pth.tar')
                if os.path.isfile(ckpt) == False:
                    shutil.rmtree(full_checkpoint, ignore_errors=True)
            else:
                rm_ckpt_all(full_checkpoint)


if __name__ == '__main__':
    rm_ckpt_all('../checkpoints')
