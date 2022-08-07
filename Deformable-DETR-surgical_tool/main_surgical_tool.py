# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

#import datasets
import util.misc as utils
#from datasets import build_dataset, get_coco_api_from_dataset
from engine_from_detr import train_one_epoch, evaluate
from models import build_model

from armory import paths

import torchvision.transforms as T
import os
import logging
from torch.utils.tensorboard import SummaryWriter

from surgical_tool_dataset import SurgicalToolDataset
from dataset_random_preprocess import random_preprocess

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    
    # dataset parameters
    parser.add_argument('--dataset_file', default='surgical_tool',  type=str,
                        choices = ['surgical_tool'])
    parser.add_argument('--dataset_path', default='./data/', type=str)
    parser.add_argument('--enable_random_resize_train', action='store_true', default=False,
                        help="We will randomly resize each sample with sampling a pool of possible size over training set.")
    parser.add_argument('--enable_random_resize_eval', action='store_true', default=False,
                        help="We will randomly resize each sample with sampling a pool of possible size over validation set. \
                        However, we do not recommend to add augmentation to your test set.")

    parser.add_argument('--enable_learn_duplicate', action='store_true', default=False,
                        help="If set true, we will use conv + bn layer to learn mapping gray scale to RGB input.")
    
    '''
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    '''


    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help = "If this tag is true, we will run the evaluation over the test set.")
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--gpu_ids', type=str, default='0')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Classfication head:
    parser.add_argument('--add_1_from_detr', action='store_true', default=False,
                        help='whether to add an extra 1 into the classification channel as following DETR setup.')

    # for debug
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode')
    parser.add_argument('--name', type=str, default="DETR_surgical_tool_train")

    # for pretrain ckpt:
    parser.add_argument('--coco_pretrain', action='store_true', default=False,
                        help='Use coco pretrain model')
    parser.add_argument('--coco_pretrain_ckpt', type=str, 
                        default=None,
                        choices=[
                                 None,
                                 "./pretrain_ckpt/r50_deformable_detr-checkpoint.pth",   # for deformable DETR_R50
                                 "./pretrain_ckpt/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth"
                        ],
                        help='The path direction to the coco pretrain ckpt.')

    # StK: we will freeze all other part but only train the classification part:
    parser.add_argument('--freeze_model', action='store_true', default=False,
                        help="We will freeze all other part but only train the classification part.")

    return parser


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids    # comment this line if running with cuda10.1
    logger = logging.getLogger(__name__)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # TODO: update build_model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.coco_pretrain_ckpt is not None:
        checkpoint = torch.load(args.coco_pretrain_ckpt, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint["model"])

    if args.coco_pretrain == True:
        # for the coco_pretrain architecture, we need to initialize the Deformable DETR with 91 classification head
        # to match with the coco pretrain checkpoint. However, when we fine-tune over the surgical tool dataset, we need
        # to change this value to match with surgical tool dataset. I.e. we need to totally change the final fc 
        new_num_classes = 4
        model_without_ddp.update_class_embed(new_num_classes=new_num_classes, device=device, add_1_from_detr=args.add_1_from_detr, with_box_refine=args.with_box_refine, two_stage=args.two_stage)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out
        
    if args.freeze_model:
        param_dicts = [
             {
                "params":
                    [p for n, p in model_without_ddp.class_embed.named_parameters() if p.requires_grad],
                "lr": args.lr,
            },
        ]
        for n, p in model_without_ddp.class_embed.named_parameters():
            if p.requires_grad:
                print('n: ',n)
                print('p: ', p)
                print()
    else:
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.dataset_file == "surgical_tool":
        
        if args.enable_random_resize_train:
            T_train = random_preprocess
        
        else:
            T_train = T.Compose([
                                T.ToTensor(),
                                T.Resize([800, 1066]),   # TBD, we can add more random shape. Check dataset folder and see what has been done over coco
                                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # we cannot include this line at here. The current input image is only
                                                                                            # gray scale image, i.e. one channel. We can only apply this normalization
                                                                                            # later when we cast from 1-channel to 3-channel sample.
                             ])
        
        if args.enable_random_resize_eval:
            # randomly resize with a pool of size seen in training time
            T_test = random_preprocess
        else:
            # default validation set shape
            T_test  = T.Compose([
                                T.ToTensor(),
                                T.Resize([800, 1066]),
                                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # we cannot include this line at here. The current input image is only
                                                                                            # gray scale image, i.e. one channel. We can only apply this normalization
                                                                                            # later when we cast from 1-channel to 3-channel sample.
                             ])
        
        train_dataset = SurgicalToolDataset("./data/", split="train", transform=T_train, 
                                            enable_random_resize=args.enable_random_resize_train, 
                                            enable_learn_duplicate=args.enable_learn_duplicate)
        
        test_dataset = SurgicalToolDataset("./data/", split="test", transform=T_test, 
                                            enable_random_resize=args.enable_random_resize_eval,  # we do not put randomly resize for the eval set
                                            enable_learn_duplicate=args.enable_learn_duplicate)
        
        
        def collate_fn(batch):
            imgs, targets = zip(*batch)
            imgs = torch.stack(imgs, dim=0)   # stack only works for tensors of same shape
            
            #imgs = torch.nested_tensor(imgs)   # this supports combing tensors of different shape together. 
                                               # See details at here: 
                                               #              https://pytorch.org/docs/master/nested.html
                                               # this function seems only support after torch 1.12
            targets = list(targets)
            return imgs, targets

        if args.enable_random_resize_train:
            # utils.collate_fn supports nested tensor
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=utils.collate_fn)
        else:
            # collate_fn does not support for nested tensor
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        
        if args.enable_random_resize_eval:
            # utils.collate_fn supports nested tensor
            # validation process only support batch size of 1
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
        else:
            # collate_fn does not support for nested tensor
            # validation process only support batch size of 1
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        
    else:
        raise NotImplementedError()

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    
    curr_best_mean_mAP = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device,  epoch, args, writer, args.clip_max_norm)

        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        evaluate_result = evaluate(
                            model, 
                            criterion, 
                            postprocessors, 
                            test_loader,
                            device,
                            args.output_dir,
                            args,
                            logger,
                            writer,
                            epoch)

        # let's save the checkpoint with the best mean mAP so far:
        mean_mAP = 0
        for key in evaluate_result['ap']:
            mean_mAP += evaluate_result['ap'][key]
        mean_mAP /= len(evaluate_result['ap'])

        if mean_mAP >= curr_best_mean_mAP:
            curr_best_mean_mAP = mean_mAP
            if args.output_dir:
                best_checkpoint_path = output_dir / 'best_checkpoint.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    }, best_checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    paths.set_mode("host")  # make the accessory of dataset to the location that armory command use
    
    main(args)
