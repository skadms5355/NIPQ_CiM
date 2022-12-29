from __future__ import print_function
from asyncio import BaseTransport
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from utils.dali_loader import imagenet_valid_from_train_loader
from utils.pascal_data import VOC_ROOT, VOCDetection
from .preproc_bi_real_net import imagenet_pca, Lighting		# for Bi-real-net
from PIL import Image
try:
    from .dali_loader import imagenet_trainloader, imagenet_validloader
except ImportError:
    pass
try:
    from pyvww.utils import VisualWakeWords
except ImportError:
    pass

__all__ = ['set_data_loader']

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# VisualWakeWords Dataset class
class VisualWakeWordsClassification(datasets.vision.VisionDataset):
    """`Visual Wake Words <https://arxiv.org/abs/1906.05721>`_ Dataset.
    Args:
        root (string): Root directory where COCO images are downloaded to.
        annFile (string): Path to json visual wake words annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, annFile, transform=None, target_transform=None, split='val'):
        self.vww = VisualWakeWords(annFile)
        self.ids = list(sorted(self.vww.imgs.keys()))
        self.split = split

        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the index of the target class.
        """

        vww = self.vww
        img_id = self.ids[index]
        ann_ids = vww.getAnnIds(imgIds=img_id)
        target = vww.loadAnns(ann_ids)[0]['category_id']
        
        path = vww.loadImgs(img_id)[0]['file_name']
        
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        
        if self.transform is not None:
            img = self.transform(img)
            
            
        if self.target_transform is not None:
            target = self.target_transform(target)
                
        return img, target
            
    def __len__(self):
        return len(self.ids)

# VWW dataloader function
def get_vww_train_valid_loader(data_dir,
                               batch_size,
                               input_size,
                               random_seed,
                               valid_size=0.,
                               shuffle=True,
                               num_workers=4,
                               distributed=False,
                               pin_memory=False):
    assert (valid_size == 0), \
            "[!] valid size should be 0 for vww dataset."

    # value extracted from training set.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # define transforms
    train_transform = transforms.Compose([
        # transforms.RandomAffine(10, translate=None, shear=(5,5,5,5), resample=False, fillcolor=0),
        transforms.RandomResizedCrop(size=(input_size,input_size), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(10, translate=None, shear=(5,5,5,5), resample=False, fillcolor=0),
        # transforms.ColorJitter(brightness=(0.6,1.4), saturation=(0.1, 1.1), hue=(-0.1, 0.1)),
        transforms.ToTensor(),
        normalize
        ])

    # load the dataset
    train_dataset = VisualWakeWordsClassification(
        root=os.path.join(data_dir, 'all2014'),
        annFile=os.path.join(data_dir, 'annotations/instances_train.json'),
        transform=train_transform, split='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn
    )

    return (train_loader, None, None)

def get_vww_test_loader(data_dir,
                        batch_size,
                        input_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=False):
    """
    Returns
    -------
    - data_loader: test set iterator.
    """

    # value extracted from trainin set.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # define transform
    test_transform = transforms.Compose([
        transforms.Resize(int(input_size / 0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
        ])

    test_dataset = VisualWakeWordsClassification(
        root=os.path.join(data_dir, 'all2014'),
        annFile=os.path.join(data_dir, 'annotations/instances_val.json'),
        transform=test_transform, split='val')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return test_loader

# Pascal dataset class
def get_pascal_train_valid_loader(dataset,
                                    data_dir,
                                    batch_size,
                                    augment,
                                    random_seed,
                                    valid_size=0.1,
                                    shuffle=True,
                                    num_workers=4,
                                    class_split=False, 
                                    per_class = 50, 
                                    distributed=False,
                                    pin_memory=False):
    assert (valid_size == 0), \
            "[!] valid size should be 0 for pascal dataset."
    
    # if augment:
        
    # else:
    #     train_transform = BaseTransport()
    
    # train_dataset = VOCDetection(root=VOC_ROOT, image_sets=[('2007', 'train'), ('2012', 'trainval')]),
    #                             transforms=


# MNIST dataloader function
def get_mnist_train_valid_loader(data_dir,
                                 batch_size,
                                 random_seed,
                                 valid_size=0.1,
                                 shuffle=True,
                                 num_workers=4,
                                 distributed=False,
                                 pin_memory=False):
    """
    If using CUDA, num_workers should be set to 1 and pin_memory to True. (why?)
    Args
    -------
        data_dir: Path directory to the dataset.
        batch_size: How many samples per batch to load.
        random_seed: Fix seed for reproducibility.
        valid_size: Percentage split of the training set used for the
                    validation set. Should be a float in the range [0, 1].
        shuffle: Whether to shuffle the train/validation indices.
        num_workers: Number of subprocesses to use when loading the dataset.
        distributed: Whether to use the DistributedSampler for loading.
        pin_memory: Whether to copy tensors into CUDA pinned memory.
                    Set it to True if using GPU.

    Returns
    -------
        train_loader: training set iterator.
        valid_loader: validation set iterator.
    """
    assert ((valid_size >= 0) and (valid_size <= 1)), \
        "[!] valid_size should be in the range [0, 1]."

    # value extracted from entire training set.
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn
    )

    if not valid_idx:
        valid_sampler = None
        valid_loader = None
    else:
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    return (train_loader, train_sampler, valid_loader)



def get_mnist_test_loader(data_dir,
                          batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=False):

    """
    Returns
    -------
    - data_loader: test set iterator.
    """

    normalize = transforms.Normalize((0.1307,), (0.3081,))

    #define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.MNIST(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


# CIFAR10 dataloader function
def get_cifar10_100_train_valid_loader(dataset,
                                       data_dir,
                                       batch_size,
                                       augment,
                                       random_seed,
                                       valid_size=0.1,
                                       shuffle=True,
                                       num_workers=4,
                                       class_split=False, 
                                       per_class = 50, 
                                       distributed=False,
                                       pin_memory=False):
    """
    If using CUDA, num_workers should be set to 1 and pin_memory to True. (why?)
    Args
    -------
        data_dir: Path directory to the dataset.
        batch_size: How many samples per batch to load.
        augment: Whether to apply the data augmentation scheme mentioned in
                 the paper. Only applied on the train split.
        random_seed: Fix seed for reproducibility.
        valid_size: Percentage split of the training set used for the
                    validation set. Should be a float in the range [0, 1].
        shuffle: Whether to shuffle the train/validation indices.
        num_workers: Number of subprocesses to use when loading the dataset.
        distributed: Whether to use the DistributedSampler for loading.
        pin_memory: Whether to copy tensors into CUDA pinned memory.
                    Set it to True if using GPU.

    Returns
    -------
        train_loader: training set iterator.
        valid_loader: validation set iterator.
    """
    assert ((valid_size >= 0) and (valid_size <= 1)), \
        "[!] valid_size should be in the range [0, 1]."

    # value extracted from entire training set.
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        if dataset == 'cifar10':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif dataset == 'cifar100':
            train_transform = transforms.Compose([
                transforms.RandomRotation((-15,15)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    if dataset == 'cifar10':
        num_class = 10
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )
    elif dataset == 'cifar100':
        num_class = 100
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if class_split:
        split_per_class = per_class
        # split_per_class = int(np.floor(split/num_class))
        remain_target = [split_per_class for i in range(0, num_class)]
        tmp_train_sampler = SequentialSampler(train_dataset)
        tmp_train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=tmp_train_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )
        counter=0
        train_idx = []
        valid_idx = []
        for batch_idx, data in enumerate(tmp_train_loader):
            inputs, targets=data
            for target in targets:
                if remain_target[target] > 0:
                    valid_idx.append(counter)
                    remain_target[target] -= 1
                else:
                    train_idx.append(counter)
                counter += 1
    else:
        train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn
    )

    if not valid_idx:
        valid_sampler = None
        valid_loader = None
    else:
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    return (train_loader, train_sampler, valid_loader)


def get_cifar10_100_test_loader(dataset,
                                data_dir,
                                batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False):

    """
    Returns
    -------
    - data_loader: test set iterator.
    """

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    #define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )
    elif dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def set_data_loader(args):

    # Data loading code
    if args.rank == 0:
        print(f"==> Preparing dataset {args.dataset}")
    if args.dataset == 'mnist':
        train_loader, train_sampler, valid_loader = get_mnist_train_valid_loader(
            data_dir=args.data, batch_size=args.train_batch,
            random_seed=42, valid_size=args.valid_size, shuffle=True,
            num_workers=args.workers, distributed=args.distributed,
            pin_memory=False)
        test_loader = get_mnist_test_loader(
            data_dir=args.data, batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':

        train_loader, train_sampler, valid_loader = get_cifar10_100_train_valid_loader(
            dataset=args.dataset, data_dir=args.data, batch_size=args.train_batch,
            augment=args.augment, random_seed=42, valid_size=args.valid_size, shuffle=True,
            num_workers=args.workers, class_split=args.class_split, per_class=args.per_class, distributed=args.distributed,
            pin_memory=False) # True for CUDA?
        test_loader = get_cifar10_100_test_loader(
            dataset=args.dataset, data_dir=args.data, batch_size=args.test_batch,
            shuffle=False, num_workers=args.workers, pin_memory=False) # True for CUDA?

    elif args.dataset == 'vww':
        train_loader, train_sampler, valid_loader = get_vww_train_valid_loader(
            data_dir=args.data, batch_size=args.train_batch, input_size=args.input_size,
            random_seed=42, valid_size=args.valid_size, shuffle=True,
            num_workers=args.workers, distributed=args.distributed,
            pin_memory=True)
        test_loader = get_vww_test_loader(
            data_dir=args.data, batch_size=args.test_batch, input_size=args.input_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)
    
    elif args.dataset == 'pascal':
        train_loader, train_sampler, valid_loader = get_pascal_train_valid_loader(
            dataset=args.dataset, data_dir=args.data, batch_size=args.train_batch,
            augment=args.augment, random_seed=42, valid_size=args.valid_size, shuffle=True,
            num_workers=args.workers, class_split=args.class_split, per_class=args.per_class, distributed=args.distributed,
            pin_memory=False)
        # test_loader = get_pascal_test_loader(
        #     data_dir=args.data, batch_size=args.test_batch, input_size=args.input_size,
        #     shuffle=False, num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        valid_sampler = None
        if args.dali:
            train_sampler = None
            train_loader = imagenet_trainloader(traindir, args)
            if args.filelist is not None:
                valid_loader = imagenet_valid_from_train_loader(traindir, args)
            else:
                valid_loader = None
            # assigned valid set to test set
            test_loader = imagenet_validloader(valdir, args)
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            if args.interpolation == 'bilinear':
                interp = Image.BILINEAR
            else:
                interp = Image.BICUBIC

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224, interpolation=interp),
                    #Lighting(0.1),                                          # for Bi-real-net
                    transforms.RandomHorizontalFlip(),
                    #transforms.ColorJitter(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            valid_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256, interpolation=interp),
#                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

            if args.distributed:
                train_sampler = DistributedSampler(train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.train_batch,
                shuffle=(False if args.distributed else True),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler, worker_init_fn=worker_init_fn)

            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=args.test_batch, shuffle=False,
#                num_workers=args.workers, pin_memory=True)
                num_workers=args.workers, pin_memory=True, sampler=valid_sampler)

            test_loader = None
    return train_loader, train_sampler, valid_loader, test_loader

