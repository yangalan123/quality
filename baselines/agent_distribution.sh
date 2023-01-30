#cd quality_data/extractive_dpr_agent_rest_20splits
tag="first"
for max_len in 300 400 500
do
    cd /data/chenghao/quality/baselines/quality_data/extractive_dpr_agent_${tag}_20splits_maxlen${max_len}
    for agent in {0..19}
    do
        mkdir agent_${agent}
        cp config.json agent_${agent}/
        cp train.jsonl.agent_${agent} agent_${agent}/train.jsonl
        cp validation.jsonl.agent_${agent} agent_${agent}/validation.jsonl
        cp test.jsonl.agent_${agent} agent_${agent}/test.jsonl
    done
done
