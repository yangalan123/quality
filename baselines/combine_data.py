import copy
from tqdm import tqdm
import os
import torch
import json
import shutil


if __name__ == '__main__':
    strategy = "concat"
    #strategy = "reverse_concat"
    #strategy = "replacement"
    #filepath_template = "/data/chenghao/quality/baselines/quality_data/extractive_rouge_full/{}.jsonl"
    filepath_template = "/data/chenghao/quality/baselines/quality_data/extractive_rouge_full/{}.jsonl"
    #output_filepath_template = "/data/chenghao/quality/baselines/quality_data/pegasus_sum_300/{}.jsonl"
    #num_agents = 5
    #num_agents = 4
    num_agents = 20
    max_length = 300
    max_length_rest = 100
    top_K = 5
    for phase in ["train", "validation", "test"]:
        for agent_i in range(num_agents):
            #output_filepath_template = f"/data/chenghao/quality/baselines/quality_data/dpr_agent_dpr_sum_300_{strategy}/agent_{agent_i}" + "/{}.jsonl"
            #output_filepath_template = f"/data/chenghao/quality/baselines/quality_data/dpr_agent_dpr_sum_combine_20splits_maxlen_300_{strategy}/agent_{agent_i}" + "/{}.jsonl"
            output_filepath_template = f"/data/chenghao/quality/baselines/quality_data/dpr_agent_dpr_sum_combine_20splits_maxlen_150_{max_length_rest}_{strategy}/agent_{agent_i}" + "/{}.jsonl"
            os.makedirs(os.path.dirname(output_filepath_template), exist_ok=True)
            shutil.copyfile("quality_data/extractive/config.json", os.path.dirname(output_filepath_template) + "/config.json")
            f_out_path = output_filepath_template.format(phase)
            flag = False
            f_out = open(f_out_path, "w", encoding='utf-8')
            #f_in_agent = f"/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent/agent_{agent_i}/{phase}.jsonl"
            #f_in_agent = f"/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent_first_20splits/agent_{agent_i}/{phase}.jsonl"
            f_in_agent = f"/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent_first_20splits_maxlen150/agent_{agent_i}/{phase}.jsonl"
            #f_in_agent_rest = f"/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent_rest/agent_{agent_i}/{phase}.jsonl"
            #f_in_agent_rest = f"/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent_rest_20splits/agent_{agent_i}/{phase}.jsonl"
            f_in_agent_rest = f"/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent_rest_20splits_maxlen{max_length_rest}/agent_{agent_i}/{phase}.jsonl"
            buf_agent = open(f_in_agent, "r", encoding='utf-8').readlines()
            buf_agent_rest = open(f_in_agent_rest, "r", encoding='utf-8').readlines()
            for line_i, line_agent in enumerate(tqdm(buf_agent)):
                line_agent_rest = buf_agent_rest[line_i]
                data_agent_rest = json.loads(line_agent_rest.strip())
                data_agent = json.loads(line_agent.strip())
                assert data_agent["query"] == data_agent_rest["query"]
                new_obj = copy.copy(data_agent)
                #tgt_text = doc_dict[data['context']]
                tgt_text = data_agent_rest['context']
                if strategy == "replacement":
                    new_obj['context'] = tgt_text
                elif "concat" in strategy:
                    if strategy == "concat":
                        new_obj['context'] += tgt_text
                    else:
                        new_obj['context'] = tgt_text + new_obj["context"]
                f_out.write(json.dumps(new_obj) + "\n")
                src_text = data_agent['context']
                if not flag:
                    #print(len(src_text[0]), "\n", src_text[:200], "\n", tgt_text)
                    print("original_sum: \n", data_agent['context'], "\nupdated_sum: \n", new_obj['context'])
                    flag = True
