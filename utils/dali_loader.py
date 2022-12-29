from __future__ import print_function

## DALI support
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    #from nvidia.dali.pipeline import Pipeline
    #import nvidia.dali.ops as ops
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError(
        "Please install DALI from \
        https://www.github.com/NVIDIA/DALI to use dali option.")

__all__ = ['imagenet_trainloader', 'imagenet_valid_from_train_loader', 'imagenet_validloader']

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, file_list=None, dali_cpu=False, interpolation=None, is_training=True):
    if file_list is not None:
        images, labels = fn.readers.file(file_root=data_dir,
                                        file_list=file_list,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
    else:
        images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    if interpolation == 'bicubic':
        interp = types.INTERP_CUBIC
    elif interpolation == 'triangular':
        interp = types.INTERP_TRIANGULAR
    elif interpolation == 'bilinear':
        interp = types.INTERP_LINEAR
    else:
        interp = types.INTERP_CUBIC
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               #random_aspect_ratio=[0.8, 1.25], # original numbers from pytorch example
                                               random_aspect_ratio=[0.75, 1.333], # what we were using
                                               #random_area=[0.1, 1.0], # original numbers from pytorch example
                                               random_area=[0.08, 1.0], # what we were using
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=interp)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=interp)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels

def imagenet_trainloader(traindir, args):
    pipe = create_dali_pipeline(batch_size=args.train_batch,
                                num_threads=args.workers,
                                device_id=args.gpu,
                                seed=12 + args.gpu,
                                data_dir=traindir,
                                crop=224,
                                size=256,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.gpu,
                                num_shards=args.world_size,
                                interpolation=args.interpolation,
                                is_training=True)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    return train_loader

def imagenet_valid_from_train_loader(traindir, args):
    pipe = create_dali_pipeline(batch_size=args.test_batch,
                                num_threads=args.workers,
                                device_id=args.gpu,
                                seed=12 + args.gpu,
                                data_dir=traindir,
                                file_list=args.filelist,
                                crop=224,
                                size=256,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.gpu,
                                num_shards=args.world_size,
                                interpolation=args.interpolation,
                                is_training=False)
    pipe.build()
    valid_loader = DALIClassificationIterator(
        pipe, reader_name="Reader")
    return valid_loader

def imagenet_validloader(valdir, args):
    pipe = create_dali_pipeline(batch_size=args.test_batch,
                                num_threads=args.workers,
                                device_id=args.gpu,
                                seed=12 + args.gpu,
                                data_dir=valdir,
                                crop=224,
                                size=256,
                                dali_cpu=args.dali_cpu,
                                shard_id=args.gpu,
                                num_shards=args.world_size,
                                interpolation=args.interpolation,
                                is_training=False)
    pipe.build()
    valid_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    return valid_loader

