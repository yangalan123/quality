import json
import os
from collections import Counter
source_dir = "quality_data/original/"
topic_dist = Counter()
articles = set()
for phase in ["train", "validation", "test"]:
    source_file = os.path.join(source_dir, "{}.jsonl".format(phase))
    output_filepath = os.path.join(source_dir, "{}.source.topic.list".format(phase))
    output_file = open(output_filepath, "w", encoding='utf-8')
    num_obj = 0
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
                obj = {"question": q, "topic": topic, "source": source}
                output_file.write(json.dumps(obj) + "\n")
                num_obj += 1
    print("write {} lines for {} phase".format(num_obj, phase))

print(json.dumps(dict(topic_dist.most_common()), indent=4))


