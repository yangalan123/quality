cur_dir="/data/chenghao/quality/baselines/huggingface_dataset/quality"
#prefix=extractive_dpr_agent_first_20splits_maxlen
#for max_len in 25 50 100 150
for flag in first rest
do
  prefix=extractive_dpr_agent_${flag}_20splits_maxlen
  for max_len in 150 300 400 500 100 50 25
  do
      tgt_dir=/data/chenghao/quality/baselines/quality_data/${prefix}${max_len}
      if [ -d "${tgt_dir}" ]; then
        cd ${tgt_dir}
        for agent in {0..19}
        do
            #mkdir agent_${agent}
            #cp config.json agent_${agent}/
            #cp train.jsonl.agent_${agent} agent_${agent}/train.jsonl
            #cp validation.jsonl.agent_${agent} agent_${agent}/validation.jsonl
            #cp test.jsonl.agent_${agent} agent_${agent}/test.jsonl
            #rmdir ${prefix}${max_len}
            mkdir -p ${cur_dir}/${prefix}${max_len}/
            python /data/chenghao/quality/baselines/huggingface_dataset/quality/book_id_finder.py ${tgt_dir}/agent_${agent}
            zip -r ${cur_dir}/${prefix}${max_len}/agent_${agent}.zip agent_${agent}
        done
      else
        echo "$tgt_dir not exist!"
      fi
  done
done
