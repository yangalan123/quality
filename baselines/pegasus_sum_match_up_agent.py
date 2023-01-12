from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import copy
from tqdm import tqdm
import os
import torch
import json


if __name__ == '__main__':
    strategy = "concat"
    #strategy = "replacement"
    #filepath_template = "/data/chenghao/quality/baselines/quality_data/extractive_rouge_full/{}.jsonl"
    filepath_template = "/data/chenghao/quality/baselines/quality_data/extractive_rouge_full/{}.jsonl"
    #output_filepath_template = "/data/chenghao/quality/baselines/quality_data/pegasus_sum_300/{}.jsonl"
    num_agents = 5
    max_length = 300
    top_K = 5

    for phase in ["train", "validation", "test"]:
        filepath = filepath_template.format(phase)
        with open(filepath, "r", encoding='utf-8') as f_in:
            buf = f_in.readlines()
            doc_set = set()
            for line in buf:
                data = json.loads(line.strip())
                doc_set.add(data['context'])
            doc_dict = dict()
            print("doc index building complete")
            try:
                print("attempt to load pre-computed summarization")
                doc_dict = torch.load(f"quality_pegasum{max_length}_{phase}.torch")
            except:
                print("loading failed, re-generate summary")
                for line in tqdm(doc_set):
                    src_text =[line, ]

                    model_name = "google/pegasus-xsum"
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    tokenizer = PegasusTokenizer.from_pretrained(model_name)
                    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
                    batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
                    translated = model.generate(**batch)
                    tgt_text = tokenizer.batch_decode(translated, max_length=max_length, skip_special_tokens=True)[0]
                    doc_dict[line] = tgt_text
                torch.save(doc_dict, f"quality_pegasum{max_length}_{phase}.torch")
                print("doc sum building complete")

        for agent_i in range(num_agents):
            output_filepath_template = f"/data/chenghao/quality/baselines/quality_data/dpr_agent_pegasus_sum_300_{strategy}/agent_{agent_i}" + "/{}.jsonl"
            os.makedirs(os.path.dirname(output_filepath_template), exist_ok=True)
            f_out_path = output_filepath_template.format(phase)
            flag = False
            f_out = open(f_out_path, "w", encoding='utf-8')
            f_in_agent = f"/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent/agent_{agent_i}/{phase}.jsonl"
            buf_agent = open(f_in_agent, "r", encoding='utf-8').readlines()
            for line_i, line_agent in enumerate(tqdm(buf_agent)):
                line = buf[line_i]
                data = json.loads(line.strip())
                data_agent = json.loads(line_agent.strip())
                new_obj = copy.copy(data_agent)
                tgt_text = doc_dict[data['context']]
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
