import os

from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import Counter
if __name__ == '__main__':
    output_dir = "visualization/stat_question_per_book"
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset("chromeNLP/quality", "dpr-first-0%-maxlen-150")
    for split in ['train', "validation", "test"]:
        counter = Counter()
        for item in ds[split]:
            counter[item['article_id']] += 1
        print(f"Now processing {split} set")
        # print("top-5")
        # print(counter.most_common(10))
        # print("least")
        # print(counter.most_common()[-10:])
        counts = counter.values()
        plt.hist(counts)
        plt.xlabel("#(questions)")
        plt.ylabel("#(books) with that many questions")
        plt.title(f"Quality Book-wise Question Distribution for {split} set")
        plt.xlim([10, 20])
        plt.savefig(os.path.join(output_dir, f"quality_{split}.pdf"))
        plt.show()
        plt.clf()
