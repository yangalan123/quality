export PYTHONPATH=/data/chenghao/quality/baselines/lrqa/:${PYTHONPATH}
export TRANSFORMERS_CACHE=/data/chenghao/transformers_cache
#EXPDIR=./experiment/cosmosqa
#EXPDIR=./experiment/extractive_dpr_roberta_base_epoch_20
#for experiment in extractive_dpr extractive_rouge oracle pegasus_sum pegasus_sum_300
#for experiment in extractive_rouge_oracle_answer extractive_rouge_no_question extractive_rouge_full
#for experiment in pegasus_sum_300_dprwiki_5  pegasus_sum_300_dprwiki_w_sum_5
#for experiment in question_only_300_dprwiki_w_sum_5
for experiment in 0 1 2 3 4
#for experiment in fiction no_fiction
do
    #echo ${experiment}
#EXPDIR=./experiment/extract_epoch_20
#EXPDIR=./experiment/extractive_dpr_oracle_roberta_base_epoch_20
#EXPDIR=./experiment/extractive_dpr_deberta_large_epoch_20
    #--task_name cosmosqa \
    #--model_name_or_path microsoft/deberta-v3-large \
    #--model_name_or_path roberta-base \
        #--task_base_path ./quality_data/extractive_dpr_2 \
        #--do_train \
        #--task_base_path ./quality_data/${experiment} \
    #base_exp_dir=extractive_dpr_agent/agent
    #base_exp_dir=dpr_agent_pegasus_sum_300_concat/agent
    base_exp_dir=dpr_agent_dpr_sum_300_concat/agent
    #base_exp_dir=pegasus_sum_300_dprwiki_5
    #base_exp_dir=pegasus_sum_300
    EXPDIR=./experiment/race_deberta_large_epoch_20/${base_exp_dir}_${experiment}
    #EXPDIR=./experiment/race_deberta_large_epoch_20/${base_exp_dir}/${experiment}
    CUDA_VISIBLE_DEVICES=0 python lrqa/run_lrqa.py \
        --model_name_or_path "./experiment/race_deberta_v3_large_epoch_20/checkpoint-best" \
        --model_mode mc \
        --task_name custom \
        --task_base_path ./quality_data/${base_exp_dir}_${experiment} \
        --learning_rate 1e-5 \
        --num_train_epochs 20 \
        --output_dir ${EXPDIR} \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --do_eval \
        --evaluation_strategy steps \
        --warmup_ratio 0.1 \
        --eval_steps 500 \
        --save_total_limit 5 \
        --overwrite_output_dir \
        --save_strategy steps \
        --load_best_model_at_end
done
