export PYTHONPATH=/data/chenghao/quality/baselines/lrqa/:${PYTHONPATH}
export TRANSFORMERS_CACHE=/data/chenghao/transformers_cache
#EXPDIR=./experiment/cosmosqa
#CUDA_VISIBLE_DEVICES=0 python pegasus_sum.py

#EXPDIR=./experiment/pegasus_sum_roberta_base_epoch_20
EXPDIR=./experiment/pegasus_sum300_dprwiki5_roberta_base_epoch_20
    #--task_name cosmosqa \
CUDA_VISIBLE_DEVICES=0,1 python lrqa/run_lrqa.py \
    --model_name_or_path roberta-base \
    --model_mode mc \
    --learning_rate 1e-5 \
    --task_name custom \
    --num_train_epochs 20 \
    --task_base_path ./quality_data/pegasus_sum_300_dprwiki_5 \
    --output_dir ${EXPDIR} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --warmup_ratio 0.1 \
    --eval_steps 500 \
    --save_total_limit 5 \
    --save_strategy steps \
    --load_best_model_at_end

python pegasus_sum_match_up_wiki.py
#EXPDIR=./experiment/pegasus_sum_300_roberta_base_epoch_20
EXPDIR=./experiment/pegasus_sum300_dprwiki5wSum_roberta_base_epoch_20
    #--task_name cosmosqa \
CUDA_VISIBLE_DEVICES=0,1 python lrqa/run_lrqa.py \
    --model_name_or_path roberta-base \
    --model_mode mc \
    --task_name custom \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --task_base_path ./quality_data/pegasus_sum_300_dprwiki_w_sum_5 \
    --output_dir ${EXPDIR} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --warmup_ratio 0.1 \
    --eval_steps 500 \
    --save_total_limit 5 \
    --save_strategy steps \
    --load_best_model_at_end
