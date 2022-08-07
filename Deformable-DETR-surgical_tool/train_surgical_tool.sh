#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
gpu_ids="6,7"
NUM_OF_GPU=2
name="Deformal_DETR_Surgical_tool_pretrain_R50_num_classes_4_detr+1_setting"
output_dir="./outputs_pretrain_deformal_detr_R50_num_classes_4_detr+1_setting"
log="log_train_R50_pretrain_num_classes_4_detr+1_setting.txt"

for split in 0 
do python -m torch.distributed.launch   \
        --nproc_per_node=${NUM_OF_GPU}  \
        --use_env                       \
        --master_port 66662             \
        main_surgical_tool.py                         \
        --lr 2e-4                       \
        --batch_size 3                  \
        --gpu_ids ${gpu_ids}            \
        --name ${name}       \
        --add_1_from_detr               \
        --coco_pretrain                   \
        --coco_pretrain_ckpt ./pretrain_ckpt/r50_deformable_detr-checkpoint.pth        \
        --output_dir ${output_dir}          \
        --debug  \
        > ${log}
    
        #CUDA_VISIBLE_DEVICES=${gpu_ids}      \
        #--coco_pretrain                   \   # this will give 91 classification head as intial architecture for loading coco-pretrain checkpoint
        #--coco_pretrain_ckpt ./pretrain_ckpt/r50_deformable_detr-checkpoint.pth \

        #--detr_demo                       \
        #--debug                                \
           #debug
           #--pretrain
           #--local_rank 0               \
           #--gradient_accumulation_steps 20  \
           #--train_batch_size 30        \
           #--eval_batch_size 30         \

#    python src/run_adv_training.py \
#           --experiment_name "${experiment_name}_adv" \
#           --fold "${split}" \
#           --resnet_path "${logdir}/fold_${split}/best_validation_acc.pth" \
#           --adv_num_epochs 1


#name=cifar10-100_500
#name="Deformal_DETR_CARLA_pretrain_R50_num_classes_5_detr+1_setting"
#output_dir="./outputs_pretrain_deformal_detr_R50_num_classes_5_detr+1_setting"
#log="log_train_R50_pretrain_num_classes_5_detr+1_setting.txt"
done