import torch
import numpy as np
from datasets import load_dataset

if __name__ == '__main__':
    example_dir = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_100_concat/agent_8/prediction/validation_predictions.p"
    prediction = torch.load(example_dir)
    ds = load_dataset("chromeNLP/quality", "dpr-first-0%-maxlen-150")
    ds_valid = ds['validation']
    bookid2perf = dict()
    for item_i, item in enumerate(ds_valid):
        answer_i = item['options'].index(item['output'])
        bookid = item['article_id']
        pred_i = int(np.argmax(prediction[item_i]))
        assert 0 <= pred_i <= 3 and len(prediction[item_i]) == 4
        if bookid not in bookid2perf:
            bookid2perf[bookid] = [0, 0]
        bookid2perf[bookid][0] += 1 if pred_i == answer_i else 0
        bookid2perf[bookid][1] += 1

    perfs = [bookid2perf[x][0] / bookid2perf[x][1] for x in bookid2perf]
    perf_mean = np.mean(perfs)
    perf_std = np.std(perfs)
    print("book-wise mean:", perf_mean)
    print("book-wise std:", perf_std)
    print("book-wise min:", np.min(perfs))
    print("book-wise max:", np.max(perfs))




