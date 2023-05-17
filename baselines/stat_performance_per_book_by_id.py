import json

import datasets
import torch
import numpy as np
from datasets import load_dataset

if __name__ == '__main__':
    #example_dir = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_100_concat/agent_8/prediction/validation_predictions.p"
    # example_dir = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/extractive_dpr_agent_first_20splits_maxlen150/agent_5/prediction/validation_predictions.p"
    #example_dir = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_random_sum_combine_20splits_maxlen_150_150_rest_full_concat/agent_5/prediction/validation_predictions.p"
    # example_dir = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_random_sum_combine_20splits_maxlen_150_150_rest_agentwise_concat/agent_5/prediction/validation_predictions.p"
    # example_dir = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_150_rest_full_concat/agent_5/prediction/validation_predictions.p"
    # example_dir = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_concat/agent_5/prediction/validation_predictions.p"
    # example_dir = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/rltuned_dpr_agent_rest_20splits_maxlen150/bookwise_30029/agent_5/prediction/validation_predictions.p"
    example_dir = "/data/chenghao/quality/baselines/experiment/t0_improved_prompt_output/first_300_qonly_150/T0pp/agent_5/validation_predictions.p"
    # example_dir = "/data/chenghao/quality/baselines/experiment/t0_improved_prompt_output/first_300_rest_fulldpr_150/T0pp/agent_5/validation_predictions.p"
    ds = load_dataset("chromeNLP/quality", "dpr-first-0%-maxlen-150")
    ds_valid = ds['validation']
    all_perfs = []
    all_perfs_train = []
    all_perfs_test = []
    for seed in range(5):
    # for seed in range(1):
        # example_dir = f"/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/rltuned_dpr_agent_rest_20splits_maxlen150/bookwise_30029_predfull/seed_{seed}/agent_5/prediction_full_book_10shot/validation_predictions.p"
        # example_dir = f"/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/rltuned_dpr_agent_rest_20splits_maxlen150/bookwise_30029_predfull/seed_{seed}/agent_0/prediction_full_book_10shot/validation_predictions.p"
        prediction = torch.load(example_dir)
        bookid2perf = dict()
        bookid2perf_train = dict()
        bookid2perf_test = dict()
        #train_dataset = datasets.load_from_disk("/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_concat/agent_5/prediction/30029_agent5.p")
        local_dataset = datasets.load_from_disk(
            f"/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_150_rest_full_concat/agent_5/prediction/30029_agent5.10shot.full.predfull.seed{seed}")
        # if "predfull" in example_dir:
        #     local_dataset = datasets.load_from_disk(f"/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_150_rest_full_concat/agent_5/prediction/30029_agent5.10shot.full.predfull.seed{seed}")
        # else:
        #     local_dataset = datasets.load_from_disk(f"/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_concat/agent_5/prediction/30029_agent5.10shot.full.seed{seed}")
        train_dataset = local_dataset.select(range(10))
        train_queries = train_dataset['question']
        # if "rltuned" in example_dir:
        #     _path = f"/data/chenghao/quality/baselines/quality_data/rltuned_dpr_agent_rest_20splits_maxlen150/bookwise_30029_predfull/seed_{seed}/agent_5/validation.jsonl"
        #     f = open(_path, "r", encoding='utf-8')
        #     buf = f.readlines()
        #     ds_valid = [json.loads(x) for x in buf]
        #     for item_i in range(len(ds_valid)):
        #         item = ds_valid[item_i]
        #         ds_valid[item_i]["options"] = [ds_valid[item_i][f'option_{x}'] for x in range(4)]
        #         ds_valid[item_i]['output'] = item['option_{}'.format(item['label'])]
        #         ds_valid[item_i]['article_id'] = "30029"
        #         ds_valid[item_i]['question'] = item['query']
        counter = 0
        for item_i, item in enumerate(ds_valid):
            answer_i = item['options'].index(item['output'])
            bookid = item['article_id']
            if bookid != "30029":
                continue
            if "rltuned" in example_dir:
                if len(prediction.shape) == 2:
                    pred_i = int(np.argmax(prediction[counter]))
                    assert 0 <= pred_i <= 3 and len(prediction[counter]) == 4
                else:
                    pred_i = prediction[counter]
            else:
                if len(prediction.shape) == 2:
                    pred_i = int(np.argmax(prediction[item_i]))
                    assert 0 <= pred_i <= 3 and len(prediction[item_i]) == 4
                else:
                    pred_i = prediction[item_i]
            if bookid not in bookid2perf:
                bookid2perf[bookid] = [0, 0]
                bookid2perf_train[bookid] = [0, 0]
                bookid2perf_test[bookid] = [0, 0]
            bookid2perf[bookid][0] += 1 if pred_i == answer_i else 0
            bookid2perf[bookid][1] += 1
            if item['question'] in train_queries:
                bookid2perf_train[bookid][0] += 1 if pred_i == answer_i else 0
                bookid2perf_train[bookid][1] += 1
            else:
                bookid2perf_test[bookid][0] += 1 if pred_i == answer_i else 0
                bookid2perf_test[bookid][1] += 1
            counter += 1

        perfs = [bookid2perf[x][0] / bookid2perf[x][1] for x in bookid2perf]
        all_perfs.append(perfs[0])
        # print(bookid2perf)
        # print(perfs[0])

        # perf_mean = np.mean(perfs)
        # perf_std = np.std(perfs)
        # print("book-wise mean:", perf_mean)
        # print("book-wise std:", perf_std)
        # print("book-wise min:", np.min(perfs))
        # print("book-wise max:", np.max(perfs))

        perfs = [bookid2perf_train[x][0] / bookid2perf_train[x][1] for x in bookid2perf_train]
        # perf_mean = np.mean(perfs)
        # perf_std = np.std(perfs)
        print(bookid2perf_train)
        # print("book-wise mean:", perf_mean)
        all_perfs_train.append(perfs[0])
        # print("book-wise std:", perf_std)
        # print("book-wise min:", np.min(perfs))
        # print("book-wise max:", np.max(perfs))

        perfs = [bookid2perf_test[x][0] / bookid2perf_test[x][1] for x in bookid2perf_test]
        all_perfs_test.append(perfs[0])
    print(f"performance overall average: {np.mean(all_perfs)} ({np.std(all_perfs)})")
    print(f"performance (train-set) average: {np.mean(all_perfs_train)} ({np.std(all_perfs_train)})")
    print(f"performance (test-set) average: {np.mean(all_perfs_test)} ({np.std(all_perfs_test)})")
    # perf_mean = np.mean(perfs)
        # perf_std = np.std(perfs)
        # print("book-wise mean:", perf_mean)
        # print("book-wise std:", perf_std)
        # print("book-wise min:", np.min(perfs))
        # print("book-wise max:", np.max(perfs))



