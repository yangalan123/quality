cd quality_data/extractive_dpr_agent_rest
for agent in 0 1 2 3 4
do
    mkdir agent_${agent}
    cp config.json agent_${agent}/
    cp train.jsonl.agent_${agent} agent_${agent}/train.jsonl
    cp validation.jsonl.agent_${agent} agent_${agent}/validation.jsonl
    cp test.jsonl.agent_${agent} agent_${agent}/test.jsonl
done
