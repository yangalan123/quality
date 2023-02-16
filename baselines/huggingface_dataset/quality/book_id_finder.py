import copy
import os.path
import json
import sys
from loguru import logger

if __name__ == '__main__':
    source_dir = "/data/chenghao/quality/baselines/quality_data/original"
    tgt_dir = sys.argv[1]
    for split in ['train', "validation", "test"]:
        source_file = os.path.join(source_dir, f"{split}.jsonl")
        map_question_option2id_difficulty = dict()
        all_items = []
        with open(source_file, "r", encoding='utf-8') as f_in:
            for line in f_in:
                instance = json.loads(line.strip())
                article_id, question_objs = instance['article_id'], instance['questions']
                for question_obj in question_objs:
                    if split != "test":
                        question, answer_id, options, difficulty = question_obj['question'], question_obj['gold_label'], question_obj['options'], question_obj['difficult']
                        # starting from 1
                        answer_id -= 1
                        gold_answer = options[answer_id]
                    else:
                        question, options, difficulty = question_obj['question'], question_obj['options'], question_obj['difficult']
                        gold_answer = "\n".join([x.strip() for x in options])

                    difficulty = "easy" if difficulty == 0 else "hard"
                    key = f"Q: {question.strip()} \n\n A: {gold_answer.strip()} \n\n"
                    # print(key)
                    # exit()
                    assert key not in map_question_option2id_difficulty, f"key conflict found: {article_id} v.s. {map_question_option2id_difficulty[key]}"
                    map_question_option2id_difficulty[key] = {"article_id": article_id, "difficulty": difficulty}
                    all_items.append([question, gold_answer, article_id, difficulty, key])
        tgt_file = os.path.join(tgt_dir, f"{split}.jsonl")
        logger.info(f"Processing {tgt_file}")
        with open(tgt_file, "r", encoding='utf-8') as f_in2:
            data = []
            for line_i, line in enumerate(f_in2):
                instance = json.loads(line.strip())
                instance_q, instance_label = instance['query'], instance['label']
                if split != "test":
                    gold_answer = instance[f"option_{instance_label}"]
                else:
                    gold_answer = "\n".join([instance[f'option_{x}'].strip() for x in range(4)])

                key = f"Q: {instance_q.strip()} \n\n A: {gold_answer.strip()} \n\n"
                if key not in map_question_option2id_difficulty:
                    import difflib
                    s1 = key
                    s2 = all_items[line_i][-1]
                    matcher = difflib.SequenceMatcher(a=s1, b=s2)
                    print("Matching Sequences:")
                    for match in matcher.get_matching_blocks():
                        print("Match             : {}".format(match))
                        print("Matching Sequence : {}".format(s1[match.a:match.a + match.size]))
                item = map_question_option2id_difficulty[key]
                new_inst = copy.deepcopy(instance)
                new_inst["article_id"] = item['article_id']
                new_inst["difficulty"] = item['difficulty']
                data.append(new_inst)
        # exit()
        assert len(data) == len(all_items)
        with open(tgt_file, "w", encoding='utf-8') as f_out2:
            for line in data:
                f_out2.write(json.dumps(line) + "\n")






