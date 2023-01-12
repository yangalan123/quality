export PYTHONPATH=/data/chenghao/quality/baselines/lrqa/:${PYTHONPATH}
export TRANSFORMERS_CACHE=/data/chenghao/transformers_cache
    #--scorer dpr \
#python lrqa/scripts/extraction.py \
    #--input_base_path ./quality_data/original \
    #--output_base_path ./quality_data/extractive_dpr \
    #--scorer dpr \
    #--query_type question
python lrqa/scripts/extraction_agent.py \
    --input_base_path ./quality_data/original \
    --output_base_path ./quality_data/extractive_dpr_agent_rest \
    --scorer dpr \
    --query_type question
