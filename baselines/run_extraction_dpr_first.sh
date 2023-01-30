export PYTHONPATH=/data/chenghao/quality/baselines/lrqa/:${PYTHONPATH}
export TRANSFORMERS_CACHE=/data/chenghao/transformers_cache
    #--scorer dpr \
#python lrqa/scripts/extraction.py \
    #--input_base_path ./quality_data/original \
    #--output_base_path ./quality_data/extractive_dpr \
    #--scorer dpr \
    #--query_type question
for max_word_count in 300 400 500
do
    CUDA_VISIBLE_DEVICES=0 python lrqa/scripts/extraction_agent.py \
        --input_base_path ./quality_data/original \
        --output_base_path ./quality_data/extractive_dpr_agent_first_20splits_maxlen${max_word_count} \
        --max_word_count ${max_word_count} \
        --scorer dpr \
        --query_type question
done
