#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
gpu_ids="5"
NUM_OF_GPU=1
#coco_pretrain_ckpt="./outputs_pretrain_deformal_detr_R50_num_classes_4_detr+1_setting/checkpoint.pth"
#log_dir="./log_eval_R50_num_classes_4_detr+1_setting_unseen"

#coco_pretrain_ckpt="./outputs_pretrain_deformal_detr_R50_num_classes_4_detr+1_setting_with_box_refine/checkpoint.pth"
#log_dir="./log_eval_R50_num_classes_4_detr+1_setting_with_box_refine_unseen"

#coco_pretrain_ckpt="./outputs_pretrain_deformal_detr_R50_num_classes_4_detr+1_setting_more_resize/checkpoint.pth"
#log_dir="./log_eval_R50_num_classes_4_detr+1_setting_more_resize_unseen"

coco_pretrain_ckpt="./outputs_pretrain_deformal_detr_R50_num_classes_4_detr+1_setting_with_box_refine_more_resize/checkpoint.pth"
log_dir="./log_eval_R50_num_classes_4_detr+1_setting_withbox_refine_more_resize_unseen"
for unseen_resize_size in "small" "medium" "large" 
do
   log="${log_dir}/${unseen_resize_size}.txt"
   python -m torch.distributed.launch   \
        --nproc_per_node=${NUM_OF_GPU}  \
        --use_env                       \
        --master_port 66665           \
        evaluate.py                         \
        --epochs 1                       \
        --gpu_ids ${gpu_ids}            \
        --add_1_from_detr               \
        --with_box_refine              \
        --enable_unseen_resize_eval     \
        --unseen_resize_size ${unseen_resize_size} \
        --name "Deformal_DETR_surgical_tool_random_resize_evaluate"       \
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