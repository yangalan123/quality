export PYTHONPATH=/data/chenghao/quality/baselines/lrqa/:${PYTHONPATH}
export TRANSFORMERS_CACHE=/data/chenghao/transformers_cache
    #--scorer dpr \
    #--query_type question
    #--output_base_path ./quality_data/extractive_rouge_oracle_answer \
python lrqa/scripts/extraction.py \
    --input_base_path ./quality_data/original \
    --output_base_path ./quality_data/extractive_rouge_no_question \
    --scorer rouge \
    --query_type oracle_answer
