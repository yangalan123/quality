import torch
import numpy as np
from datasets import load_dataset

if __name__ == '__main__':
    example_dir = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_random_sum_combine_20splits_maxlen_150_150_rest_full_concat/agent_5/prediction/validation_predictions.p"
    example_dir2 = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_random_sum_combine_20splits_maxlen_150_150_rest_agentwise_concat/agent_5/prediction/validation_predictions.p"
    example_dir3 = "/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_150_rest_full_concat/agent_5/prediction/validation_predictions.p"
    example_dirs = [example_dir, example_dir2, example_dir3]
    ds = load_dataset("chromeNLP/quality", "dpr-first-0%-maxlen-150")['validation']

    for i in range(3):
        for j in range(i+1, 3):
            data = torch.load(example_dirs[i])
            data_argmax = np.argmax(data, axis=-1)
            data2 = torch.load(example_dirs[j])
            data2_argmax = np.argmax(data2, axis=-1)
            print(type(data))
            if (data == data2).all():
                print(f"dist equal: \n{example_dirs[i]}\n{example_dirs[j]}")
            else:
                diff = sum(data != data2)
                print(f"dist non-equal {diff} / {len(data)}: \n{example_dirs[i]}\n{example_dirs[j]}")

            if (data_argmax == data2_argmax).all():
                print(f"argmax equal: \n{example_dirs[i]}\n{example_dirs[j]}")
            else:
                book_counter = 0
                for item_i in range(len(data)):
                    if data_argmax[item_i] != data2_argmax[item_i]:
                        if ds[item_i]['article_id'] == "30029":
                            book_counter += 1
                diff = sum(data_argmax != data2_argmax)
                print(f"argmax non-equal {diff} / {len(data)} ({book_counter} / 19): \n{example_dirs[i]}\n{example_dirs[j]}")

