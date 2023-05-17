# code file to transform original QuALITY dataset to HF style, with some preprocessing
# the main difference between this code and non-original version is here we use the original full context
# that can be helpful if later we want to use LongT5 or other summarizer to collect data for RM training or PPO
import os
import random

import torch
from datasets import load_dataset
from lrqa.preproc.extraction_agent import get_sent_data
import sys

MAX_WORD_COUNT = 300
# MAX_WORD_COUNT = 150
RANDOM_TIMES = 1000
RANDOM_PER_Q = 1
AGENT_I = 5
AGENT_KNOWLEDGE_LENGTH = int(sys.argv[2])
num_agents = 20
REST_FLAG = sys.argv[1]


# REST_FLAG = "full"

def process_data(example, query_focused_format=True):
    ret = {
        "context": [],
        "article_id": [],
        "difficulty": [],
        "question": [],
        "options": [],
        "output": [],
        "original": [],
        # original, random-create, k-means created
    }
    tmp_example = dict()
    for key in example:
        assert len(example[key]) == 1
    tmp_example["question"] = example['query'][0]
    tmp_example['options'] = [example[f'option_{i}'][0] for i in range(4)]
    tmp_example['output'] = tmp_example["options"][example['label'][0]]
    tmp_example["article_id"] = example['article_id'][0]
    tmp_example["difficulty"] = example['difficulty'][0]
    # article_id = example['article_id']
    # assert article_id in overall_book_sum_dict
    # sum_pool = random.sample(overall_book_sum_dict[article_id], RANDOM_PER_Q)
    # sent_data = get_sent_data(example['context'][0])
    # if AGENT_I > 0:
    #     sent_ids = list(range(len(sent_data)))
    #     # random.shuffle(sent_ids)
    #     num_per_split = len(sent_ids) // num_agents
    #     if REST_FLAG == "rest":
    #         # agent-knowledge + rest-doc, assuming oracle access to agent knowledge range
    #         sent_data = sent_data[AGENT_I * num_per_split:]
    #
    # sum_pool = []
    # for rand_i in range(RANDOM_PER_Q):
    #     random.shuffle(sent_data)
    #     cur_counter = 0
    #     sum_ret = []
    #     for sent_i in range(len(sent_data)):
    #         if cur_counter + sent_data[sent_i]["word_count"] <= MAX_WORD_COUNT:
    #             sum_ret.append(sent_data[sent_i]['text'])
    #         else:
    #             break
    #     sum_ret = " ".join(sum_ret)
    #     if AGENT_I > 0 and "agent_knowledge" in example:
    #         sum_ret = example["agent_knowledge"][0] + sum_ret
    #     sum_pool.append(sum_ret)

    # for key in ret:
    #     if key != "original":
    #         ret[key].append(example[key])
    #     else:
    #         ret[key].append("original")
    sum_pool = []
    #sum_pool.append(example['context'][0])
    # scrolls format
    _options = []
    for _option, _tag in zip(tmp_example['options'], ["A", "B", "C", "D"]):
        _options.append("({}) {}".format(_tag, _option))
    if query_focused_format:
        sum_pool.append(("{}\n\n{}\n\n{}").format(tmp_example['question'], "\n\n".join(_options), example['context'][0]))
    else:
        sum_pool.append(example['context'][0])

    for ctx_i in sum_pool:
        for key in ret:
            if key == "context":
                ret[key].append(ctx_i)
                continue
            if key != "original":
                ret[key].append(tmp_example[key])
            else:
                ret[key].append("original")

    # return example
    return ret


def create_summary(example, random_times=1000, max_word_count=150):
    # the number of random times should be much larger than necessarily required
    # ideally, we should resample per instance, but that is a huge waste of comp res,
    # so we can create a pool of random summaries, and every time pick out some of it
    article_id = example['article_id']
    sent_data = get_sent_data(example['context'])
    global overall_book_sum_dict
    if article_id not in overall_book_sum_dict:
        overall_book_sum_dict[article_id] = []
        for rand_i in range(random_times):
            random.shuffle(sent_data)
            cur_counter = 0
            ret = []
            for sent_i in range(len(sent_data)):
                if cur_counter + sent_data[sent_i]["word_count"] <= max_word_count:
                    ret.append(sent_data[sent_i]['text'])
                else:
                    break
            ret = " ".join(ret)
            overall_book_sum_dict[article_id].append(ret)
    return example


if __name__ == '__main__':
    random.seed(1234)
    root_dir = f"/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent_first_20splits_maxlen{AGENT_KNOWLEDGE_LENGTH}/agent_{AGENT_I}/" + "{}.jsonl"
    data_files = {
        "train": root_dir.format("train"),
        "validation": root_dir.format("validation"),
        # "test": root_dir.format("test")
    }
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir="/data/chenghao/hf_home")
    root_dir = "/data/chenghao/quality/baselines/huggingface_dataset/backup/original/{}.jsonl"
    data_files = {
        "train": root_dir.format("train"),
        "validation": root_dir.format("validation"),
        # "test": root_dir.format("test")
    }
    raw_datasets_original = load_dataset("json", data_files=data_files, cache_dir="/data/chenghao/hf_home")

    for split in data_files.keys():
        raw_datasets_original[split] = raw_datasets_original[split].add_column("article_id",
                                                                               raw_datasets[split]['article_id'])
        raw_datasets_original[split] = raw_datasets_original[split].add_column("difficulty",
                                                                               raw_datasets[split]['difficulty'])
        raw_datasets_original[split] = raw_datasets_original[split].add_column("agent_knowledge",
                                                                               raw_datasets[split]['context'])

    sum_dict_path = f"/data/chenghao/quality/baselines/quality_data/overall_book_sum_dict_rand{RANDOM_TIMES}_maxlen_{MAX_WORD_COUNT}.pth"
    try:
        overall_book_sum_dict = torch.load(sum_dict_path)
    except:
        overall_book_sum_dict = {}
        raw_datasets_original.map(create_summary, num_proc=4)
        torch.save(overall_book_sum_dict, sum_dict_path)

    raw_datasets_updated = raw_datasets_original.map(
        lambda example: process_data(example, query_focused_format=False),
        remove_columns=['query', "label", "agent_knowledge"] + [f'option_{i}' for i in range(4)],
        num_proc=20,
        batched=True,
        batch_size=1
    )
    agent_flag = "agent_{}_maxlen_{}_{}".format(AGENT_I, AGENT_KNOWLEDGE_LENGTH,
                                                REST_FLAG)
    # save_dir = f"/cs/scratch/chenghao/quality_data/random_summary_{agent_flag}_rand_{RANDOM_PER_Q}_{MAX_WORD_COUNT}"
    #save_dir = f"/cs/scratch/chenghao/quality_data/query_focused_original_scrolls_format"
    save_dir = f"/cs/scratch/chenghao/quality_data/original_format"
    os.makedirs(save_dir, exist_ok=True)
    for split in data_files.keys():
        raw_datasets_updated[split].save_to_disk(save_dir + f"/{split}.ds")
    # for split in data_files.keys():
    #     print(raw_datasets[split][0].keys())
    #     print(raw_datasets[split][0])
    #     print(type(raw_datasets))
    # print(raw_datasets[split][0].keys())
