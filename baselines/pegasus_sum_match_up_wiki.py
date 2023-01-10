from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import copy
from tqdm import tqdm, trange
import os
import torch
import json
import editdistance as edist


if __name__ == '__main__':
    max_length = 300
    top_K = 5
    filepath_template = "/data/chenghao/quality/baselines/quality_data/extractive_rouge_full/{}.jsonl"
    ####
    #question_check = True
    #output_filepath_template = f"/data/chenghao/quality/baselines/quality_data/pegasus_sum_{max_length}_dprwiki_{top_K}"
    #matchup_dir = "/data/chenghao/efficientqa_QA_data/quality_retrieval/{}_output"
    ####
    #question_check = False
    #output_filepath_template = f"/data/chenghao/quality/baselines/quality_data/pegasus_sum_{max_length}_dprwiki_w_sum_{top_K}"
    #matchup_dir = "/data/chenghao/efficientqa_QA_data/quality_pegasum300_retrieval/{}_output"
    ####
    ####
    question_check = False
    output_filepath_template = f"/data/chenghao/quality/baselines/quality_data/question_only_{max_length}_dprwiki_w_sum_{top_K}"
    matchup_dir = "/data/chenghao/efficientqa_QA_data/quality_pegasum300_retrieval/{}_output"
    ####
    output_filepath_template += "/{}.jsonl"
    os.makedirs(os.path.dirname(output_filepath_template), exist_ok=True)
    phase_mapping = {"train": "train", "validation":"dev"}

    for phase in ["train", "validation", "test"]:
        if phase in phase_mapping:
            matchup_file = matchup_dir.format(phase_mapping[phase])
            matchup_obj = json.load(open(matchup_file, "r"))
        else:
            matchup_obj = []

        filepath = filepath_template.format(phase)
        f_out_path = output_filepath_template.format(phase)
        flag = False
        f_out = open(f_out_path, "w", encoding='utf-8')
        with open(filepath, "r", encoding='utf-8') as f_in:
            buf = f_in.readlines()
            doc_set = set()
            for buf_i, line in enumerate(buf):
                data = json.loads(line.strip())
                doc_set.add(data['context'])
                if len(matchup_obj) > 0 and question_check:
                    replaced_query = data['query'].replace("\n", "")
                    remove_space_query = "".join(replaced_query.strip().split())
                    remove_space2query = "".join(matchup_obj[buf_i]['question'].strip().split())
                    #assert "".join(replaced_query.strip().split()) == "".join(matchup_obj[buf_i]['question'].strip().split()), f"prob=\n{remove_space_query}\n{remove_space2query}"
                    assert edist.eval(remove_space_query, remove_space2query) <= 2, f"prob=\n{remove_space_query}\n{remove_space2query}"
                    #assert data['query'] == matchup_obj[buf_i]["question"] or data['query'].replace("\n", "") == matchup_obj[buf_i]['question'], f"original_query={data['query']},\nreplaced_query={replaced_query}\nnew_query={matchup_obj[buf_i]['question']}"
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

            for line_i, line in enumerate(tqdm(buf)):
                data = json.loads(line.strip())
                new_obj = copy.copy(data)
                #tgt_text = doc_dict[data['context']]
                tgt_text = ""
                if len(matchup_obj) > 0:
                    tgt_text += "\n".join([x['title'] + ":" + x['text'] for x in matchup_obj[line_i]['ctxs'][:top_K]])
                new_obj['context'] = tgt_text
                f_out.write(json.dumps(new_obj) + "\n")
                src_text = data['context']
                if not flag:
                    print(len(src_text[0]), "\n", src_text[0][:200], "\n", tgt_text)
                    flag = True
                # src_text = [
                #     """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
                # ]
                # assert (
                #     tgt_text[0]
                #     == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
                # )
        f_out.close()
