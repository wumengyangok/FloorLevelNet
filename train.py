"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
from scipy import ndimage


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepV3PlusHANet',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='FLN')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=4,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--snapshot_pe', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--hanet', nargs='*', type=int, default=[0,1,1,1,1],
                    help='Row driven attention networks module')
parser.add_argument('--hanet_set', nargs='*', type=int, default=[3, 64, 1],
                    help='Row driven attention networks module')
parser.add_argument('--hanet_pos', nargs='*', type=int, default=[0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--pos_rfactor', type=int, default=8,
                    help='number of position information, if 0, do not use')
parser.add_argument('--aux_loss', action='store_true', default=False,
                    help='auxilliary loss on intermediate feature map')
parser.add_argument('--attention_loss', type=float, default=0.5)
parser.add_argument('--hanet_poly_exp', type=float, default=0.0)
parser.add_argument('--backbone_lr', type=float, default=0.0,
                    help='different learning rate on backbone network')
parser.add_argument('--hanet_lr', type=float, default=0.0,
                    help='different learning rate on attention module')
parser.add_argument('--hanet_wd', type=float, default=0.0001,
                    help='different weight decay on attention module')                    
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--pos_noise', type=float, default=0.0)
parser.add_argument('--no_pos_dataset', action='store_true', default=False,
                    help='get dataset with position information')
parser.add_argument('--use_hanet', action='store_true', default=True,
                    help='use hanet')
parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')

vis_map = {
        # 0: (0, 255, 0),
        0: (255, 0, 0),  # left plane
        1: (0, 101, 255),  # right plane
        2: (0, 255, 101),  # front plane
        3: (205, 255, 0),  # ground
        4: (205, 0, 255),  # sidewalk
    }
plane_map_inverse = {
        1: [0, 0, 255],  # left plane
        2: [255, 101, 0],  # right plane
        3: [101, 255, 0],  # front plane
        4: [0, 255, 205],  # ground
        5: [0, 255, 205],  # sidewalk
    }
plane_map = {
	1: [255,   0,   0], # left plane
	2: [  0, 101, 255], # right plane
	3: [  0, 255, 101], # front plane
	4: [205, 255,   0], # ground
	5: [205,   0, 255], # sidewalk
}

args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

# if args.apex:
# Check that we are running with cuda as distributed is only supported for cuda.
torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

# torch.distributed.init_process_group(backend='nccl',
#                                         init_method=args.dist_url,
#                                         world_size=args.world_size, rank=args.local_rank)

from datasets.FLNloader import FLNDataset

def inference():

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    real_data = FLNDataset(csv_file=None, phase='real')
    real_loader = DataLoader(real_data, batch_size=1, shuffle=True, num_workers=0)

    criterion, criterion_val = loss.get_loss(args)
    if args.aux_loss:
        criterion_aux = loss.get_loss_aux(args)
        net = network.get_net(args, criterion, criterion_aux)
    else:
        net = network.get_net(args, criterion)

    for i in range(5):
        if args.hanet[i] > 0:
            args.use_hanet = True

    if (args.use_hanet and args.hanet_lr > 0.0):
        optim, scheduler, optim_at, scheduler_at = optimizer.get_optimizer_attention(args, net)
    else:
        optim, scheduler = optimizer.get_optimizer(args, net)

    epoch = 0
    i = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()

    model_dir = "models"
    configs = "DLV3+HAL_batch{}_allplane".format(4)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, configs)
    net = torch.load(model_path)
    run_real_cases(real_loader, net, writer)


def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    if args.attention_loss>0 and args.hanet[4]==0:
        print("last hanet is not defined !!!!")
        exit()

    train_data = FLNDataset(csv_file=None, phase='train')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
    val_data = FLNDataset(csv_file=None, phase='val')
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)

    real_data = FLNDataset(csv_file=None, phase='real')
    real_loader = DataLoader(real_data, batch_size=1, shuffle=True, num_workers=0)

    criterion, criterion_val = loss.get_loss(args)
    if args.aux_loss:
        criterion_aux = loss.get_loss_aux(args)
        net = network.get_net(args, criterion, criterion_aux)
    else:
        net = network.get_net(args, criterion)

    for i in range(5):
        if args.hanet[i] > 0:
            args.use_hanet = True

    if (args.use_hanet and args.hanet_lr > 0.0):
        optim, scheduler, optim_at, scheduler_at = optimizer.get_optimizer_attention(args, net)
    else:
        optim, scheduler = optimizer.get_optimizer(args, net)

    epoch = 0
    i = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()

    model_dir = "models"
    configs = "DLV3+HAL_batch{}_allplane".format(4)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    while i < args.max_iter:
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        if (args.use_hanet and args.hanet_lr > 0.0):
            i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter, optim_at, scheduler_at)
            train_loader.sampler.set_epoch(epoch + 1)
        else:
            i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter)
            train_loader.sampler.set_epoch(epoch + 1)
        epoch += 1
        torch.save(net, os.path.join(model_dir, configs))


def train(train_loader, net, optim, curr_epoch, writer, scheduler, max_iter, optim_at=None, scheduler_at=None):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    requires_attention = False

    if args.attention_loss > 0:
        get_attention_gt = Generate_Attention_GT()
        criterion_attention = loss.get_loss_bcelogit(args)
        requires_attention = True
    
    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):

        if curr_iter >= max_iter:
            break
        start_ts = time.time()

        inputs, gts, line_gts, _img_name, aux_gts = data['X'], data['l'], data['ll'], data['PA'], data['ll']

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        aux_gts = np.asarray(aux_gts)

        B, N, H = 4, 13, 16
        attention_labels = torch.zeros(B, N, H)
        ostride = aux_gts.shape[1] // H
        # threads = []
        for j in range(0, aux_gts.shape[0]):
            for k in range(0, N):
                rows = np.where(aux_gts[j] == k)[0]
                if len(rows) > 0:
                    row = np.unique((rows // ostride), return_counts=False)
                    # print("channel", k, "row", row)
                    attention_labels[j][k][row] = 1
        attention_labels = attention_labels.cuda()
        inputs, gts, line_gts = inputs.cuda(), gts.cuda(), line_gts.cuda()

        optim.zero_grad()
        if optim_at is not None:
            optim_at.zero_grad()

        outputs = net(inputs, gts=line_gts, gts_y=gts, aux_gts=None, attention_loss=requires_attention, attention_map=False)

        main_loss = outputs[0]
        attention_map = outputs[1]
        attention_labels = get_attention_gt(aux_gts, attention_map.shape)
        attention_labels = attention_labels.cuda()
        attention_loss = criterion_attention(input=attention_map.transpose(1,2), target=attention_labels.transpose(1,2))
        #del inputs, gts, aux_gts
        total_loss = main_loss + (args.attention_loss * attention_loss)
        total_loss.backward()
        optim.step()
        if optim_at is not None:
            optim_at.step()
        scheduler.step()
        if scheduler_at is not None:
            scheduler_at.step()
        time_meter.update(time.time() - start_ts)
        # del total_loss, log_total_loss
        print(total_loss.data)
        curr_iter += 1

        if args.local_rank == 0:
            if i % 50 == 49:
                if optim_at is not None:
                    print('finish 50')
                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [lr_at {:0.6f}], [time {:0.4f}]'.format(
                    curr_epoch, i + 1, len(train_loader), curr_iter, total_loss.avg,
                    optim.param_groups[-1]['lr'], optim_at.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)
                else:
                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, 0,
                        optim.param_groups[-1]['lr'], time_meter.avg / 4)
                print(msg)

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter

def run_real_cases(real_loader, net, writer):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    for real_idx, data in enumerate(real_loader):
        inputs, img_name,  = data['X'], data['PA']
        print(img_name)
        with torch.no_grad():
            output, attention_map, output_y = net(inputs.cuda(), attention_map=True)
        print(output_y.shape)
        print(attention_map)
        predictions = output.data.max(1)[1].cpu()
        planes = output_y.data.max(1)[1].cpu()
        print(predictions.shape)
        predictions = np.asarray(predictions[0], dtype=np.uint8)
        planes = np.asarray(planes[0], dtype=np.uint8)
        # show(predictions)
        im = cv2.imread(img_name[0])
        # continue
        im = post_process(im, predictions, planes)
        im = cv2.resize(im, dsize=(360, 480))

        cv2.imwrite(img_name[0][:-4] + '_result.png', im)

        # cv2.imwrite()



def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    try:
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
    except:
        pass
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    # cv2.imshow("Biggest component", img2)
    # cv2.waitKey()
    return img2

def show(im):
    cv2.imshow('', im)
    cv2.waitKey()

def denoise(img):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 50
    # your answer image
    img2 = np.zeros((output.shape), dtype=np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1
    return img2

def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def intersectionPoint(l1, l2):
    a1 = np.array([l1[0][0], l1[0][1]])
    a2 = np.array([l1[1][0], l1[1][1]])
    b1 = np.array([l2[0][0], l2[0][1]])
    b2 = np.array([l2[1][0], l2[1][1]])

    a = a2 - a1
    b = b2 - b1
    a_perp = perp(a)
    denom = np.dot(a_perp, b)
    if denom == 0:
        return None  # intersect in infinity
    num = np.dot(a_perp, a1 - b1)
    result = (num / denom) * b + b1
    return [int(result[0]), int(result[1])]


# def rgb2ind(im, color_map=plane_map):
# 	"""
# 	Convert rgb plane map into index image based on the index<->rgb value mapping
# 		defined in plane_map
#
# 	Args:
# 		im: input image with shape = [h, w, 3]
# 		color_map: mapping of the index and rgb value, default use plane_map
# 	Returns:
# 		index image with shape = [h, w]
# 	"""
#
# 	ind = np.zeros((im.shape[0], im.shape[1]))
# 	for i, rgb in color_map.items(): # use iteritems() instead when using Python2
# 		ind[(im == rgb).all(2)] = i
# 	return ind.astype(np.uint8)


def validate_lines(this_line, l_lines_per_plane):
    if len(l_lines_per_plane) == 0:
        return True
    mid_p = [0.5 * (this_line[0][0] + this_line[1][0]), 0.5 * (this_line[0][1] + this_line[1][1])]
    # print(mid_p)
    for line in l_lines_per_plane:
        p = intersectionPoint(line, this_line)
        # print(p)
        # print(line)
        if p is None: continue
        if p[0] > min(line[0][0], line[1][0]) and p[0] < max(line[0][0], line[1][0]):
            # print('0')
            return False
        if p[1] > min(line[0][1], line[1][1]) and p[1] < max(line[0][1], line[1][1]):
            # print('1')
            return False
        p = [0.5 * (line[0][0] + line[1][0]), 0.5 * (line[0][1] + line[1][1])]
        # print(p)
        if mid_p[1] > p[1]: return False
    return True


def post_process(im, line, planes):
    im = cv2.resize(im, dsize=(320, 320))
    im = np.array(im)
    planes = np.array(planes)
    print(planes.shape)
    segment = planes

    # show(segment * 50)
    # if plane_type

    for plane_type in [3]:
        # show((segment == plane_type).astype(np.uint8) * 255)
        # continue
        labeled, nr_objects = ndimage.label((segment == plane_type).astype(np.uint8))
        # for id in range(nr_objects + 1):
        mask = ((segment == plane_type).astype(np.uint8)).astype(np.uint8)
        if np.sum(mask) > np.sum(mask * (segment == plane_type)):
            continue
        if np.sum(mask) < 5000:
            continue
        # show(mask * 255)
        idx = np.where(mask == 1)
        # original_shape = im.shape
        # im = cv2.resize(line, dsize=(480, 360))
        # cluster_frame = cv2.imread(video_dir + file_name + '_level.png', cv2.IMREAD_GRAYSCALE) // 25
        # range_frame =
        # Find the left/right range
        left_most = np.min(idx[1], axis=0)
        right_most = np.max(idx[1], axis=0)
        center = (left_most + right_most) // 2
        # Set the length of line segment
        fixed_length = 100
        zero = np.zeros(shape=(320, 320), dtype=np.uint8)
        zero[:, left_most: right_most] = 1
        mask = zero
        cluster_mask = line * mask
        # show(cluster_mask * 25)
        # continue
        num_floor = np.max(cluster_mask)
        # regress line per floor
        ini_k = []
        simple_line = []
        level_point_list = []
        l_lines_per_plane = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 1

        for floor in range(1, num_floor + 1):
            one_floor = (cluster_mask == floor).astype(np.uint8)
            # show(one_floor * 255)
            one_floor = denoise(one_floor)
            # give up too small floor region
            if np.sum(one_floor) < 500:
                continue
            # Choose the largest floor region
            # show(one_floor * 255)
            # continue
            one_floor = undesired_objects(one_floor)
            # show(one_floor * 255)
            idx = np.where(one_floor > 0)
            # Find left/right range (overwrite the global l/r range here... )
            # left_most = np.min(idx[1], axis=0)
            # right_most = np.max(idx[1], axis=0)
            # Collect point set {(x,y)}
            x, y = np.where(one_floor > 0)
            # x = np.expand_dims(x)
            # point_list = np.concatenate([np.expand_dims(x, -1),np.expand_dims(y, -1)], axis=-1)
            point_list = 1000 * x + y
            if len(point_list) == 0:
                continue
            rand_list = np.random.choice(point_list, size=50, replace=True)
            # print(rand_list)
            rand_list = [(n % 1000, n // 1000) for n in rand_list]
            level_point_list.append([rand_list, floor])
            x = [p[0] for p in rand_list]
            y = [p[1] for p in rand_list]
            params = np.polyfit(x, y, deg=1)
            f = np.poly1d(params)
            params = np.polyfit(x, y, deg=1)
            f = np.poly1d(params)
            k = f(1.) - f(0.)
            ini_k.append(k)
            simple_line.append([[left_most, f(left_most)], [right_most, f(right_most)]])
            this_line = [[left_most, int(f(left_most))], [right_most, int(f(right_most))], floor]
            if validate_lines(this_line, l_lines_per_plane):
                l_lines_per_plane.append(this_line)
                cv2.line(im, pt1=(left_most, int(f(left_most))),
                         pt2=(right_most, int(f(right_most))),
                         thickness=5, color=vis_map[floor % 5])
                check_mask = np.zeros(shape=(320, 320))
                cv2.line(check_mask, pt1=(left_most, int(f(left_most))),
                         pt2=(right_most, int(f(right_most))),
                         thickness=1, color=1)
                cv2.putText(im, str(floor),
                            ((left_most + right_most) // 2, int(f((left_most + right_most) // 2))),
                            font,
                            fontScale, vis_map[floor % 5], 2, cv2.LINE_AA)

            ''' uncommit following lines for vp constrainted optimization '''
            # if len(simple_line) < 2:
            #     continue
            # [x, y] = intersectionPoint(simple_line[-1], simple_line[0])
            # ini_k.append(x)
            # ini_k.append(y)
            # print(ini_k)
            # import scipy.optimize as opt
            #
            # # ‘Nelder - Mead’
            # # ‘Powell’ (see here
            # # ‘CG’
            # # ‘BFGS’
            # # ‘Newton - CG’
            # # ‘L - BFGS - B’
            # # ‘TNC’
            # # ‘COBYLA’
            # # ‘SLSQP’
            # # ‘trust - constr
            # # ‘dogleg’
            # # ‘trust - ncg’
            # # ‘trust - exact’
            # # ‘trust - krylov’
            # def demo_func(x):
            #     # print(x)
            #     xv = x[0]
            #     # print(xv)
            #     yv = x[1]
            #     # print(yv)
            #     energy = 0.
            #     k = []
            #     for level in range(len(level_point_list)):
            #         # print(x[level])
            #         rand_list = level_point_list[level][0]
            #         x = [p[0] for p in rand_list]
            #         y = [p[1] for p in rand_list]
            #         weight = [1 for p in rand_list]
            #         x.append(xv)
            #         y.append(yv)
            #         weight.append(10)
            #         params = np.polyfit(x, y, deg=1, w=weight, full=True)
            #         energy += params[1]
            #
            #     return energy
            # result = opt.minimize(demo_func, x0=[x, y])
            # # sa = SA(func=demo_func, x0=[240, 180], T_max=1, T_min=1e-9, L=300, max_stay_counter=300)
            # print(result)
            # best_x, best_y = result.x[0], result.x[1]
            # print('best_x:', best_x, 'best_y', best_y)
            #
            # for level in range(len(level_point_list)):
            #     vpx = best_x
            #     vpy = best_y
            #     rand_list = level_point_list[level][0]
            #     floor = level_point_list[level][1]
            #     x = [p[0] for p in rand_list]
            #     y = [p[1] for p in rand_list]
            #     weight = [1 for p in rand_list]
            #     x.append(vpx)
            #     y.append(vpy)
            #     weight.append(10)
            #     params = np.polyfit(x, y, deg=1, w=weight, full=True)
            #     f = np.poly1d(params[0])
            #     this_line = [[left_most, int(f(left_most))], [right_most, int(f(right_most))], floor]
            #     if validate(this_line, l_lines_per_plane):
            #         l_lines_per_plane.append(this_line)
            #         cv2.line(im, pt1=(left_most, int(f(left_most))), pt2=(right_most, int(f(right_most))),
            #                  thickness=5, color=vis_map[floor % 5])
            #         check_mask = np.zeros(shape=(320, 320))
            #         cv2.line(check_mask, pt1=(left_most, int(f(left_most))),
            #                  pt2=(right_most, int(f(right_most))),
            #                  thickness=1, color=1)
            #         cv2.putText(im, str(floor),
            #                     ((left_most + right_most) // 2, int(f((left_most + right_most) // 2))),
            #                     font,
            #                     fontScale, vis_map[floor % 5], 2, cv2.LINE_AA)
            #         cv2.line(im, pt1=(left_most, int(f(left_most))),
            #                  pt2=(right_most, int(f(right_most))),
            #                  thickness=2, color=vis_map[floor % 5])
    return im




def validate(val_loader, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, optim_at=None, scheduler_at=None):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        if args.no_pos_dataset:
            inputs, gt_image, img_names = data
        elif args.pos_rfactor > 0:
            inputs, gt_image, img_names, _, (pos_h, pos_w) = data
        else:
            inputs, gt_image, img_names, _ = data

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            if args.pos_rfactor > 0:
                if args.use_hanet and args.hanet_pos[0] > 0:  # use hanet and position
                    output, attention_map, pos_map = net(inputs, pos=(pos_h, pos_w), attention_map=True)
                else:
                    output = net(inputs, pos=(pos_h, pos_w))
            else:
                output = net(inputs)

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == args.dataset_cls.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             args.dataset_cls.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        if optim_at is not None:
            evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                        writer, curr_epoch, args.dataset_cls, curr_iter, optim_at, scheduler_at)
        else:
            evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                        writer, curr_epoch, args.dataset_cls, curr_iter)

    return val_loss.avg

num_vis_pos = 0

def visualize_pos(writer, pos_maps, iteration):
    global num_vis_pos
    #if num_vis_pos % 5 == 0:
    #    save_pos_numpy(pos_maps, iteration)
    num_vis_pos += 1

    stage = 'valid'
    for i in range(len(pos_maps)):
        pos_map = pos_maps[i]
        if isinstance(pos_map, tuple):
            num_pos = 2
        else:
            num_pos = 1

        for j in range(num_pos):
            if num_pos == 2:
                pos_embedding = pos_map[j]
            else:
                pos_embedding = pos_map

            H, D = pos_embedding.shape
            pos_embedding = pos_embedding.unsqueeze(0)  # 1 X H X D
            if H > D:   # e.g. 32 X 8
                pos_embedding = F.interpolate(pos_embedding, H, mode='nearest') # 1 X 32 X 8
                D = H
            elif H < D:   # H < D, e.g. 32 X 64
                pos_embedding = F.interpolate(pos_embedding.transpose(1,2), D, mode='nearest').transpose(1,2) # 1 X 32 X 64
                H = D
            if args.hanet_pos[1]==1: # pos encoding
                pos_embedding = torch.cat((torch.ones(1, H, D).cuda(), pos_embedding/2, pos_embedding/2), 0)
            else:   # pos embedding
                pos_embedding = torch.cat((torch.ones(1, H, D).cuda(), torch.sigmoid(pos_embedding*20),
                                        torch.sigmoid(pos_embedding*20)), 0)
            pos_embedding = vutils.make_grid(pos_embedding, padding=5, normalize=False, range=(0,1))
            writer.add_image(stage + '/Pos/layer-' + str(i) + '-' + str(j), pos_embedding, iteration)

def save_pos_numpy(pos_maps, iteration):
    file_fullpath = '/home/userA/shchoi/Projects/visualization/pos_data/'
    file_name = str(args.date) + '_' + str(args.hanet_pos[0]) + '_' + str(args.exp) + '_layer'

    for i in range(len(pos_maps)):
        pos_map = pos_maps[i]
        if isinstance(pos_map, tuple):
            num_pos = 2
        else:
            num_pos = 1

        for j in range(num_pos):
            if num_pos == 2:
                pos_embedding = pos_map[j]
            else:
                pos_embedding = pos_map

            H, D = pos_embedding.shape
            pos_embedding = pos_embedding.data.cpu().numpy()   # H X D
            file_name_post = str(i) + '_' + str(j) + '_' + str(H) + 'X' + str(D) + '_' + str(iteration)
            np.save(file_fullpath + file_name + file_name_post, pos_embedding)
import cv2



class Generate_Attention_GT(object):   # 34818
    def __init__(self, n_classes=13):
        self.channel_weight_factor = 0   # TBD
        self.ostride = 0
        self.labels = 0
        self.attention_labels = 0
        self.n_classes = n_classes

    def rows_hasclass(self, B, C):
        rows = np.where(self.labels[B]==C)[0]
        if len(rows) > 0:
            row = np.unique((rows//self.ostride), return_counts=False)
            print("channel", C, "row", row)
            self.attention_labels[B][C][row] = 1

    def __call__(self, labels, attention_size):
        B, C, H = attention_size
        print(labels.shape, attention_size)
        self.labels = np.asarray(labels)
        # print(self.n_classes)
        # print(B)
        # print(H)
        self.attention_labels = torch.zeros(B, self.n_classes, H)
        self.ostride = labels.shape[1] // H
        # threads = []
        for j in range(0, labels.shape[0]):
            for k in range(0, self.n_classes):
                rows = np.where(self.labels[j]==k)[0]
                if len(rows) > 0:
                    row = np.unique((rows//self.ostride), return_counts=False)
                    # print("channel", k, "row", row)
                    self.attention_labels[j][k][row] = 1
        return self.attention_labels


if __name__ == '__main__':
    main()
    # inference()
