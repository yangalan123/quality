import json
import os
from collections import Counter
source_dir = "quality_data/original/"
#target_dir = "quality_data/pegasus_sum_300_dprwiki_5/"
target_dir = "quality_data/pegasus_sum_300/"
target_dir_decomp_no_fiction = f"{target_dir}no_fiction"
target_dir_decomp_fiction = f"{target_dir}fiction"
os.makedirs(target_dir_decomp_no_fiction, exist_ok=True)
os.makedirs(target_dir_decomp_fiction, exist_ok=True)
topic_dist = Counter()
articles = set()
for phase in ["train", "validation", "test"]:
    source_file = os.path.join(source_dir, "{}.jsonl".format(phase))
    target_file = open(os.path.join(target_dir, "{}.jsonl".format(phase)), "r", encoding='utf-8')
    target_file_0 = open(os.path.join(target_dir_decomp_no_fiction, "{}.jsonl".format(phase)), "w", encoding='utf-8')
    target_file_1 = open(os.path.join(target_dir_decomp_fiction, "{}.jsonl".format(phase)), "w", encoding='utf-8')
    buf = target_file.readlines()

    #output_filepath = os.path.join(source_dir, "{}.source.topic.list".format(phase))
    #output_file = open(output_filepath, "w", encoding='utf-8')
    num_obj = 0
    counter = 0
    with open(source_file, "r", encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line.strip())
            questions = data['questions']
            topic = data['topic']
            source = data['source']
            if data["article_id"] not in articles:
                topic_dist[topic] += 1
                articles.add(data['article_id'])
            for q in questions:
                topic_key = topic.lower()
                if "fiction" in topic_key:
                    target_file_1.write(buf[counter])
                else:
                    target_file_0.write(buf[counter])
                counter += 1

                #obj = {"question": q, "topic": topic, "source": source}
                #output_file.write(json.dumps(obj) + "\n")
                #num_obj += 1
    print("write {} lines for {} phase".format(num_obj, phase))
