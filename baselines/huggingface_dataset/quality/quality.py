import json
import itertools
import os
import datasets
from datasets import DatasetInfo

_URL = ""
_QUALITY_CITATION = """
@inproceedings{pang-etal-2022-quality,
    title = "{Q}u{ALITY}: Question Answering with Long Input Texts, Yes!",
    author = "Pang, Richard Yuanzhe  and
      Parrish, Alicia  and
      Joshi, Nitish  and
      Nangia, Nikita  and
      Phang, Jason  and
      Chen, Angelica  and
      Padmakumar, Vishakh  and
      Ma, Johnny  and
      Thompson, Jana  and
      He, He  and
      Bowman, Samuel",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.391",
    pages = "5336--5358",
    abstract = "To enable building and testing models on long-document comprehension, we introduce QuALITY, a multiple-choice QA dataset with context passages in English that have an average length of about 5,000 tokens, much longer than typical current models can process. Unlike in prior work with passages, our questions are written and validated by contributors who have read the entire passage, rather than relying on summaries or excerpts. In addition, only half of the questions are answerable by annotators working under tight time constraints, indicating that skimming and simple search are not enough to consistently perform well. Our baseline models perform poorly on this task (55.4{\%}) and significantly lag behind human performance (93.5{\%}).",
}
"""


class QualityConfig(datasets.BuilderConfig):
    def __init__(self, features, data_url, citation, label_classes=("0", "1", "2", "3"), **kwargs):
        super(QualityConfig, self).__init__(version=datasets.Version("0.0.1"), **kwargs)
        self.features = features
        self.data_url = data_url
        self.label_classes = label_classes
        self.citation = citation


class Quality(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = list(itertools.chain.from_iterable([
        [
            QualityConfig(
                name=f"dpr-first-{x * 5}%-maxlen-{maxlen}",
                features=['question', "context", "output", "article_id", "difficulty"],
                data_url=f"extractive_dpr_agent_first_20splits_maxlen{maxlen}/agent_{x}.zip",
                citation=_QUALITY_CITATION,
                description=f"Using DPR (NYU-version) to summarize first {x * 5}% of the document within {maxlen} max tokens"
            ) for x in range(0, 20)
        ] for maxlen in [150, 300, 400, 500]
  ])) + list(itertools.chain.from_iterable([
        [
            QualityConfig(
                name=f"dpr-rest-{x * 5}%-maxlen-{maxlen}",
                features=['question', "context", "output", "article_id", "difficulty"],
                data_url=f"extractive_dpr_agent_rest_20splits_maxlen{maxlen}/agent_{x}.zip",
                citation=_QUALITY_CITATION,
                description=f"Using DPR (NYU-version) to summarize rest {x * 5}% of the document within {maxlen} max tokens"
            ) for x in range(0, 20)
        ] for maxlen in [25, 50, 100, 150, 300, 400, 500]
    ]))
  #+ [
            #QualityConfig(
                #name=f"original",
                #features=['question', "context", "output"],
                #data_url=f"original/original.zip",
                #citation=_QUALITY_CITATION,
                #description=f"original QuALITY data"
            #),
        #]

    # + [
    #     QualityConfig(
    #         name=f"dpr-rest-{x}%-maxlen-25",
    #         features=['question', "context", "option"],
    #         data_url=_URL + f"/extractive_dpr_agent_first_20splits_maxlen25/agent_{x}/data.zip",
    #         citation=_QUALITY_CITATION,
    #         description=f"Using DPR (NYU-version) to summarize rest {x}% of the document within 150 max tokens"
    #     ) for x in range(0, 20)
    # ]

    def _info(self) -> DatasetInfo:
        features = {feature: datasets.Value("string") for feature in self.config.features}
        features['options'] = datasets.Sequence(datasets.Value("string"))
        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(features),
            citation=_QUALITY_CITATION
        )
    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(self.config.data_url) or ""
        task_name = _get_task_name_from_data_url(self.config.data_url)
        dl_dir = os.path.join(dl_dir, task_name)
        #if self.config.name in ["axb", "axg"]:
            #return [
                #datasets.SplitGenerator(
                    #name=datasets.Split.TEST,
                    #gen_kwargs={
                        #"data_file": os.path.join(dl_dir, f"{task_name}.jsonl"),
                        #"split": datasets.Split.TEST,
                    #},
                #),
            #]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "train.jsonl"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "validation.jsonl"),
                    "split": datasets.Split.VALIDATION,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "test.jsonl"),
                    "split": datasets.Split.TEST,
                },
            ),
        ]
    def _generate_examples(self, data_file, split):
        with open(data_file, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                question = row["query"]
                context = row['context']
                options = [row[f"option_{i}"] for i in range(4)]
                label = options[row['label']]
                article_id = row["article_id"]
                difficulty = row['difficulty']
                yield f"{self.config.name}-{split}-{idx}", {
                    "context": context,
                    "output": label,
                    "options": options,
                    "question": question,
                    "article_id": article_id,
                    "difficulty": difficulty
                }


def _get_task_name_from_data_url(data_url):
    #setup = data_url.split("/")[-2]
    #agent = data_url.split("/")[-1].split("agent_")[-1]
    return data_url.split("/")[-1].split(".")[0]
    #first_flag = "first" in setup
    #maxlen = setup.split("maxlen")[-1]
    #return f"dpr-{'first' if first_flag else 'rest'}-{agent}%-maxlen-{maxlen}%"

