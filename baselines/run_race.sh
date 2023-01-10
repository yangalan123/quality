export PYTHONPATH=/data/chenghao/quality/baselines/lrqa/:${PYTHONPATH}
export TRANSFORMERS_CACHE=/data/chenghao/transformers_cache
#EXPDIR=./experiment/cosmosqa
EXPDIR=./experiment/extractive_rouge_roberta_large_epoch_20
EXPDIR=./experiment/extractive_rouge_oracle_answer_roberta_base_epoch_20
    #--task_base_path ./quality_data/extractive_rouge_oracle_answer \
EXPDIR=./experiment/extractive_rouge_full_roberta_base_epoch_20
    #--task_base_path ./quality_data/extractive_rouge_full \
EXPDIR=./experiment/extractive_rouge_no_question_base_epoch_20
    #--task_base_path ./quality_data/extractive_rouge_no_question \
EXPDIR=./experiment/race_deberta_v3_large_epoch_20
    #--task_name cosmosqa \
CUDA_VISIBLE_DEVICES=0,1 python lrqa/run_lrqa.py \
    --model_name_or_path microsoft/deberta-v3-large \
    --model_mode mc \
    --task_name race \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --output_dir ${EXPDIR} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --warmup_ratio 0.1 \
    --eval_steps 500 \
    --save_total_limit 5 \
    --overwrite_output_dir \
    --save_strategy steps \
    --load_best_model_at_end
