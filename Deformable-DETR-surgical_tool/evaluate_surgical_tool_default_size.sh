#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
gpu_ids="1"
NUM_OF_GPU=1
#coco_pretrain_ckpt="./outputs_pretrain_deformal_detr_R50_num_classes_4_detr+1_setting/checkpoint.pth"
#log="./log_eval_R50_num_classes_4_detr+1_setting_default.txt"

#coco_pretrain_ckpt="./outputs_pretrain_deformal_detr_R50_num_classes_4_detr+1_setting_with_box_refine/checkpoint.pth"
#log="./log_eval_R50_num_classes_4_detr+1_setting_with_box_refine_default.txt"

#coco_pretrain_ckpt="./outputs_pretrain_deformal_detr_R50_num_classes_4_detr+1_setting_more_resize/checkpoint.pth"
#log="./log_eval_R50_num_classes_4_detr+1_setting_more_resize_default.txt"

coco_pretrain_ckpt="./outputs_pretrain_deformal_detr_R50_num_classes_4_detr+1_setting_with_box_refine_more_resize/checkpoint.pth"
log="./log_eval_R50_num_classes_4_detr+1_setting_withbox_refine_more_resize_default.txt"
for split in 0 
do
   python -m torch.distributed.launch   \
        --nproc_per_node=${NUM_OF_GPU}  \
        --use_env                       \
        --master_port 66661            \
        evaluate.py                         \
        --epochs 1                       \
        --gpu_ids ${gpu_ids}            \
        --add_1_from_detr               \
        --with_box_refine              \
        --name "Deformal_DETR_surgical_tool_default_evaluate"       \
        --coco_pretrain_ckpt ${coco_pretrain_ckpt}\
        > ${log}
        
        #--with_box_refine              \
        
        
        #--debug                           \

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
done