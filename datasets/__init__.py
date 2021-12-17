"""
Dataset setup and loaders
"""
from datasets import cityscapes

import torchvision.transforms as standard_transforms

import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader
import torch

def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    if args.dataset == 'FLN':
        args.dataset_cls = cityscapes
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
    else:
        raise Exception('Dataset {} is not supported'.format(args.dataset))

    # Readjust batch size to mini-batch size for syncbn
    if args.syncbn:
        args.train_batch_size = args.bs_mult
        args.val_batch_size = args.bs_mult_val

    
    args.num_workers = 1 * args.ngpu
    if args.test_mode:
        args.num_workers = 1


    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Geometric image transformations
    train_joint_transform_list = []

    train_joint_transform_list += [
        joint_transforms.RandomSizeAndCrop(args.crop_size,
                                           crop_nopad=args.crop_nopad,
                                           pre_size=args.pre_size,
                                           scale_min=args.scale_min,
                                           scale_max=args.scale_max,
                                           ignore_index=args.dataset_cls.ignore_label),
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]

    if args.rrotate > 0:
        train_joint_transform_list += [joint_transforms.RandomRotate(
            degree=args.rrotate,
            ignore_index=args.dataset_cls.ignore_label)]

    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # Image appearance transformations
    train_input_transform = []
    if args.color_aug:
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=args.color_aug,
            contrast=args.color_aug,
            saturation=args.color_aug,
            hue=args.color_aug)]

    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]
    else:
        pass



    train_input_transform += [standard_transforms.ToTensor(),
                              standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()
    
    if args.jointwtborder: 
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(args.dataset_cls.ignore_label, 
            args.dataset_cls.num_classes)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    target_aux_train_transform = extended_transforms.MaskToTensor()

    if args.dataset == 'FLN':
        city_mode = args.city_mode #'train' ## Can be trainval
        city_quality = 'fine'
        if args.class_uniform_pct:
            if args.coarse_boost_classes:
                coarse_boost_classes = \
                    [int(c) for c in args.coarse_boost_classes.split(',')]
            else:
                coarse_boost_classes = None
            
            if args.pos_rfactor > 0:
                train_set = args.dataset_cls.CityScapesUniformWithPos(
                    city_quality, city_mode, args.maxSkip,
                    joint_transform_list=train_joint_transform_list,
                    transform=train_input_transform,
                    target_transform=target_train_transform,
                    target_aux_transform=target_aux_train_transform,
                    dump_images=args.dump_augmentation_images,
                    cv_split=args.cv,
                    class_uniform_pct=args.class_uniform_pct,
                    class_uniform_tile=args.class_uniform_tile,
                    test=args.test_mode,
                    coarse_boost_classes=coarse_boost_classes,
                    pos_rfactor=args.pos_rfactor)
            else:
                train_set = args.dataset_cls.CityScapesUniform(
                    city_quality, city_mode, args.maxSkip,
                    joint_transform_list=train_joint_transform_list,
                    transform=train_input_transform,
                    target_transform=target_train_transform,
                    target_aux_transform=target_aux_train_transform,
                    dump_images=args.dump_augmentation_images,
                    cv_split=args.cv,
                    class_uniform_pct=args.class_uniform_pct,
                    class_uniform_tile=args.class_uniform_tile,
                    test=args.test_mode,
                    coarse_boost_classes=coarse_boost_classes)
        else:
            if args.pos_rfactor > 0:
                train_set = args.dataset_cls.CityScapesWithPos(
                    city_quality, city_mode, 0, 
                    joint_transform=train_joint_transform,
                    transform=train_input_transform,
                    target_transform=target_train_transform,
                    target_aux_transform=target_aux_train_transform,
                    dump_images=args.dump_augmentation_images,
                    cv_split=args.cv,
                    pos_rfactor=args.pos_rfactor)
            else:
                train_set = args.dataset_cls.CityScapes(
                    city_quality, city_mode, 0, 
                    joint_transform=train_joint_transform,
                    transform=train_input_transform,
                    target_transform=target_train_transform,
                    target_aux_transform=target_aux_train_transform,
                    dump_images=args.dump_augmentation_images,
                    cv_split=args.cv)

        if args.pos_rfactor > 0:
            val_set = args.dataset_cls.CityScapesWithPos('fine', 'val', 0, 
                                                transform=val_input_transform,
                                                target_transform=target_transform,
                                                cv_split=args.cv,
                                                pos_rfactor=args.pos_rfactor)
        else:
            val_set = args.dataset_cls.CityScapes('fine', 'val', 0, 
                                                transform=val_input_transform,
                                                target_transform=target_transform,
                                                cv_split=args.cv)
    else:
        raise Exception('Dataset {} is not supported'.format(args.dataset))
    
    if args.syncbn:
        from datasets.sampler import DistributedSampler
        train_sampler = DistributedSampler(train_set, pad=True, permutation=True, consecutive_sample=False)
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)

    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True, sampler = train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , shuffle=False, drop_last=False, sampler = val_sampler)

    return train_loader, val_loader,  train_set


