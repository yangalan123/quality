import json
import os.path

if __name__ == '__main__':
    dir1 = "/data/chenghao/quality/baselines/quality_data/extractive_dpr"
    dir2 = "/data/chenghao/quality/baselines/quality_data/extractive_dpr_agent_rest_20splits_maxlen300/agent_0"
    for phase in ['train', 'validation', 'test']:
        f1 = open(os.path.join(dir1, f"{phase}.jsonl"), "r", encoding='utf-8')
        f2 = open(os.path.join(dir2, f"{phase}.jsonl"), "r", encoding='utf-8')
        buf1 = f1.readlines()
        buf2 = f2.readlines()
        assert len(buf1) == len(buf2)
        for i in range(len(buf1)):
            data1 = json.loads(buf1[i])
            data2 = json.loads(buf2[i])
            assert data1['context'] == data2['context'] and data1['query'] == data2['query']



