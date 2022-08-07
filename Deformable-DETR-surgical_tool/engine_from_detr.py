# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils

#TODO: put metric from armory here for evaluate
#DONE: modify criterion
#DONE: modify cocoevaluator content

from evaluation_metrics import EvaluationMetrics
from util import box_ops

import numpy as np
#from mean_average_precision import MetricBuilder
import torchvision.transforms as T

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, 
                    args, writer, max_norm = 0):
    print('Start training for current epoch.', flush=True)
    '''
        Inputs:
            model: torch.nn.Module, 
            criterion: torch.nn.Module,
            data_loader: Iterable, 
            optimizer: torch.optim.Optimizer,
            device: torch.device, 
            epoch: int,
            max_norm: float = 0,
            args: argument from main function
    '''
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    count = 0 # track the amount of samples in the clean dataset we have used
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if args.debug == True:
            if count >= 2:
                break
        samples = samples.to(device)
        
        if args.dataset_file == "surgical_tool" \
           and args.enable_random_resize_train == False \
           and args.enable_learn_duplicate == False:
            # this following torch.repeat func does not support for nested ternsor

            # original input is grayscale. We need to expand to dim of 3:
            samples = samples.repeat([1, 3, 1, 1])
            samples = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(samples)

            # TODO: we can consider of using a conv layer to expand the dim and train the conv layer. 
            # However, if we use a convolutional layer, then the normalization from ImageNet cannot be
            # put into effect.

        # targets shape for surgical tool dataset: batch size x dict {key: value}
        # the "boxes" label are 0 - 1 scale and is cxcywh
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        if args.debug:
            print("output len: ", outputs["pred_logits"].shape)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        writer.add_scalar("train/loss", scalar_value=loss_value, 
                            global_step=epoch * (len(data_loader))+ count)
        
        count += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger, flush=True)
    print("finish training for this epoch", flush=True)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(model, criterion, postprocessors, data_loader,
            device, output_dir, args, logger, writer, epoch):
    '''
       Input:
            postprocessors: cast the output from detr to the output used for CARLA evaluation
                        (see PostProcess class in detr.py file)
            data_loader: Iterable, 
            args:  argument from main function
    '''
    logger.info("***** Running Validation *****")
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test over training set split:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    
    evaluator = EvaluationMetrics()
    
    logger.info("\n")
    logger.info("Over test set:")
    logger.info("  Num steps = %d", len(data_loader))
    logger.info("  Batch size = %d", 1)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test over surgical tool test set:'
    
    count = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        if args.debug == True:
            if count >= 2:
                break

        samples = samples.to(device)

        if args.dataset_file == "surgical_tool"   \
            and args.enable_random_resize_eval == False   \
            and args.enable_learn_duplicate == False:
            # this following torch.repeat func does not support for nested ternsor

            # original input is grayscale. We need to expand to dim of 3:
            samples = samples.repeat([1, 3, 1, 1])
            samples = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(samples)
            
            # TODO: we can consider of using a conv layer to expand the dim and train the conv layer. 
            # However, if we use a convolutional layer, then the normalization from ImageNet cannot be
            # put into effect.

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        writer.add_scalar("test/loss", scalar_value=loss_value, 
                            global_step=epoch * (len(data_loader))+ count)

        if epoch % 1 == 0 and epoch != -1:
            orig_target_sizes = torch.unsqueeze(targets[0]["orig_size"], dim=0).repeat([len(targets), 1]).to(device) # shape of batch_size x 2 to contain [height, width]
            # return of list [{'scores':,'labels':, 'boxes':}]
            # boxes are xyxy in actual image width and length scale
            processed_output = postprocessors['bbox'](outputs, orig_target_sizes) 
            
            # we need to cast targets' boxes into the format of xyxy and in actual image width and length scale:
            targets_np = []
            for target in targets:
                reform_boxes = box_ops.box_cxcywh_to_xyxy(target["boxes"])
                img_h, img_w = target["orig_size"]
                reform_boxes[..., 0] *= img_w
                reform_boxes[..., 1] *= img_h
                reform_boxes[..., 2] *= img_w
                reform_boxes[..., 3] *= img_h
                
                target_np = {
                              "boxes": reform_boxes.cpu().numpy(),
                              "labels": target["labels"].cpu().numpy()
                             }
                targets_np.append(target_np)

            evaluator.update_eval(targets_np, processed_output) #check the format of outputs

        count += 1


    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger, flush=True)

    evaluate_result = evaluator.access_record_all()

    if epoch % 1 == 0 and epoch != -1:
        print("devygL: ",evaluate_result['ap'])
        logger.info("Validation Results over test set:")
        logger.info("AP_per_class: 0: %2.5f, 1: %2.5f, 2:  %2.5f, 3:  %2.5f" % (evaluate_result['ap'][0],
                                                                                evaluate_result['ap'][1], 
                                                                    evaluate_result['ap'][2], 
                                                                    evaluate_result['ap'][3]))
        logger.info("disappearance_rate: %2.5f" % (evaluate_result['dr']))
        logger.info("hallucinations_per_image: %2.5f" % (evaluate_result['hpi']))
        logger.info("misclassification_rate: %2.5f" % (evaluate_result['mr']))
        logger.info("true_positive_rate: %2.5f" % (evaluate_result['tpr']))

        print("\n Validation Results over test set:")
        print("AP_per_class: 0: {:2.5f}, 1: {:2.5f}, 2:  {:2.5f}, 3:  {:2.5f}".format(
            evaluate_result['ap'][0],
            evaluate_result['ap'][1], 
                                                                    evaluate_result['ap'][2], 
                                                                    evaluate_result['ap'][3]), flush=True)
        print("disappearance_rate: {:2.5f}".format(evaluate_result['dr']), flush=True)
        print("hallucinations_per_image: {:2.5f}".format(evaluate_result['hpi']), flush=True)
        print("misclassification_rate: {:2.5f}".format(evaluate_result['mr']), flush=True)
        print("true_positive_rate: {:2.5f}".format(evaluate_result['tpr']), flush=True)

        writer.add_scalar("test/AP_per_class_0:", scalar_value=evaluate_result['ap'][0], 
                                global_step=epoch)
        writer.add_scalar("test/AP_per_class_1:", scalar_value=evaluate_result['ap'][1], 
                                global_step=epoch)
        writer.add_scalar("test/AP_per_class_2:", scalar_value=evaluate_result['ap'][2], 
                                global_step=epoch)
        writer.add_scalar("test/AP_per_class_3:", scalar_value=evaluate_result['ap'][3], 
                                global_step=epoch)
        writer.add_scalar("test/disappearance_rate:", scalar_value=evaluate_result['dr'], 
                                global_step=epoch)
        writer.add_scalar("test/hallucinations_per_image:", scalar_value=evaluate_result['hpi'], 
                                global_step=epoch)
        writer.add_scalar("test/misclassification_rate:", scalar_value=evaluate_result['mr'], 
                                global_step=epoch)
        writer.add_scalar("test/true_positive_rate:", scalar_value=evaluate_result['tpr'], 
                                global_step=epoch)
        
        return evaluate_result

        

