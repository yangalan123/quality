from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import copy
from tqdm import tqdm
import os
import torch
import json
from Summarization import Summarizer
from lrqa.preproc.extraction import get_sent_data


if __name__ == '__main__':
    strategy = "concat"
    #strategy = "replacement"
    #filepath_template = "/data/chenghao/quality/baselines/quality_data/extractive_rouge_full/{}.jsonl"
    filepath_template = "/data/chenghao/quality/baselines/quality_data/extractive_rouge_full/{}.jsonl"
    #output_filepath_template = "/data/chenghao/quality/baselines/quality_data/pegasus_sum_300/{}.jsonl"
    num_agents = 5
    max_length = 300
    top_K = 5
    summarizer = Summarizer()
    save_flag = False

    for phase in ["train", "validation", "test"]:
        filepath = filepath_template.format(phase)
        with open(filepath, "r", encoding='utf-8') as f_in:
            buf = f_in.readlines()
            raw_data = [json.loads(x.strip()) for x in buf]
            doc_dict = dict()
            try:
                print("attempt to load pre-computed summarization")
                doc_dict = torch.load(f"quality_pegasum_agent_{num_agents}_maxlen_{max_length}_{phase}.torch")
            except:
                print("loading failed, re-generate summary")
                save_flag = True

        for agent_i in tqdm(range(num_agents), position=0, leave=False, desc=f"Running Agent-wise Split Sum for {phase}"):
            # output_filepath_template = f"/data/chenghao/quality/baselines/quality_data/dpr_agent_pegasus_sum_300_{strategy}/agent_{agent_i}" + "/{}.jsonl"
            output_filepath_template = f"/data/chenghao/quality/baselines/quality_data/dpr_agent_pegasus_sum_rest_{max_length}_{strategy}/agent_{agent_i}" + "/{}.jsonl"
            os.makedirs(os.path.dirname(output_filepath_template), exist_ok=True)
            f_out_path = output_filepath_template.format(phase)
            flag = False
            f_out = open(f_out_path, "w", encoding='utf-8')
            f_in_agent = f"/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent/agent_{agent_i}/{phase}.jsonl"
            buf_agent = open(f_in_agent, "r", encoding='utf-8').readlines()
            for line_i, line_agent in enumerate(tqdm(buf_agent, leave=False, position=1, desc="Processing Files")):
                # line = buf[line_i]
                # data = json.loads(line.strip())
                data = raw_data[line_i]["context"]
                sent_data = get_sent_data(data, clean_text=True)
                sent_ids = list(range(len(sent_data)))
                num_per_split = len(sent_ids) // num_agents
                # starting from 0, no need to +1
                cur_split_ids = sent_ids[(agent_i) * num_per_split: ]
                cur_context = " ".join([sent_data[x]["text"] for x in cur_split_ids])
                if cur_context not in doc_dict:
                    if not save_flag:
                        raise ValueError("inconsistent save_flag setting: cur_context not in the pre-computed file")
                    doc_dict[cur_context] = summarizer([cur_context, ])
                data_agent = json.loads(line_agent.strip())
                new_obj = copy.copy(data_agent)
                # tgt_text = doc_dict[data['context']]
                tgt_text = doc_dict[cur_context]
                if strategy == "replacement":
                    new_obj['context'] = tgt_text
                elif strategy == "concat":
                    new_obj['context'] += tgt_text
                f_out.write(json.dumps(new_obj) + "\n")
                src_text = data_agent['context']
                if not flag:
                    #print(len(src_text[0]), "\n", src_text[:200], "\n", tgt_text)
                    print(data_agent['context'], "\n", new_obj['context'])
                    flag = True
                    # src_text = [
                    #     """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
                    # ]
                    # assert (
                    #     tgt_text[0]
                    #     == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
                    # )
            f_out.close()

        if save_flag:
            torch.save(doc_dict, f"quality_pegasum_agent_{num_agents}_maxlen_{max_length}_{phase}.torch")
