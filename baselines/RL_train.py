import os.path
import numpy as np

from lrqa.preproc.extraction import get_sent_data
from PragmaticsUtils import DPRScorer
from datasets import load_dataset
import torch
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from dataclasses import dataclass, field
from lrqa.utils.model_tweaks import adjust_tokenizer
from transformers.file_utils import PaddingStrategy
from typing import Optional
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.data.data_collator import default_data_collator
import lrqa.tasks as tasks
from lrqa.utils.hf_utils import parse_args, last_checkpoint_handling
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from loguru import logger
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """
#
#     model_name_or_path: str = field(
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
#     )
#     model_mode: str = field(
#         metadata={"help": "{mc,generation,encoder-decoder}"}
#     )
#     config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
#     )
#     use_fast_tokenizer: bool = field(
#         default=True,
#         metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
#     )
#     model_revision: str = field(
#         default="main",
#         metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
#     )
#     max_seq_length: int = field(
#         default=256,
#         metadata={
#             "help": "The maximum total input sequence length after tokenization. Sequences longer "
#                     "than this will be truncated, sequences shorter will be padded."
#         },
#     )
#     padding_strategy: PaddingStrategy = field(
#         default="max_length",
#         metadata={
#             "help": "Whether to pad all samples to `max_seq_length`. "
#                     "If False, will pad the samples dynamically when batching to the maximum length in the batch."
#         },
#     )
#     parallelize: bool = field(
#         default=False,
#         metadata={
#             "help": "Whether to parallelize the model."
#         }
#     )
#     truncation_strategy: TruncationStrategy = field(
#         default="only_first",
#         metadata={
#             "help": "Whether to pad all samples to `max_seq_length`. "
#                     "If False, will pad the samples dynamically when batching to the maximum length in the batch."
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
#     )
#     torch_dtype_fp16: bool = field(
#         default=False,
#         metadata={"help": "Enable this and set model_revision='fp16' for fp16 GPT-J"},
#     )
#     eval_phase: str = field(
#         default="validation",
#         metadata={"help": "Phase for evaluation (train|validation|test)"},
#     )
#     predict_phases: str = field(
#         default="test",
#         metadata={"help": "Comma separated phases for evaluation (train|validation|test)"},
#     )
#
#     def __post_init__(self):
#         self.padding_strategy = PaddingStrategy(self.padding_strategy)
#         self.truncation_strategy = TruncationStrategy(self.truncation_strategy)


if __name__ == '__main__':
    # model_args, task_args, training_args = parse_args(HfArgumentParser((
    #     ModelArguments,
    #     tasks.TaskArguments,
    #     TrainingArguments,
    # )))
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast_tokenizer,
    #     revision=model_args.model_revision,
    # )
    # adjust_tokenizer(tokenizer)
    scorer = DPRScorer(device="cuda:0")
    # ideally, they should be controlled via arguments. I will add that later
    NUM_AGENTS = 20
    MAXLEN = 25
    BATCH_SIZE = 20
    LEARNING_RATE = 2e-5
    loss = torch.nn.BCELoss()
    logger.add("running_rl_log.txt")
    for agent_i in range(0, NUM_AGENTS, 5):
        logger.debug(f"Now doing training for {agent_i}")
        optimizer = Adam(scorer.parameters(), lr=LEARNING_RATE)
        # using training subset to avoid using validation data
        # the deberta model is trained on RACE, so it should be fine
        dataset = load_dataset("chromeNLP/quality", f"dpr-rest-{agent_i * 5}%-maxlen-{MAXLEN}")['train']
        prediction_dir = f"/data/chenghao/quality/baselines/experiment/race_deberta_large_epoch_20/dpr_agent_dpr_sum_combine_20splits_maxlen_150_{MAXLEN}_concat/agent_{agent_i}/prediction"
        predictions_logits = torch.load(os.path.join(prediction_dir, "train_predictions.p"))
        predictions = np.argmax(predictions_logits, axis=-1)
        answers = np.array([x['options'].index(x['output']) for x in dataset])
        pragmatics_labels = torch.from_numpy(predictions == answers).int().tolist()
        dataset = dataset.add_column("label", pragmatics_labels)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        counter = 0
        _loss_values = 0
        for batch in dataloader:
            optimizer.zero_grad()
            _batch_scores = []
            questions = batch['question']
            contexts = batch['context']
            labels = batch['label']
            questions_embeds = scorer.embed(questions, "question")
            contexts_embeds = scorer.embed(contexts, "context")
            scores = -torch.norm(questions_embeds.unsqueeze(-2) - contexts_embeds, dim=-1)
            exp_scores = torch.exp(scores)
            normalized_scores = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)
            _loss = loss(torch.diag(normalized_scores), labels.float().to(scorer.device))
            _loss_values += _loss.cpu().item()
            _loss.backward()
            optimizer.step()
            counter += len(questions)
            if counter % (5 * BATCH_SIZE) == 0:
                logger.info(f"Agent-{agent_i}-current progress: {counter}/{len(dataset)}, _loss: {_loss_values / counter * BATCH_SIZE}")
        scorer.context_encoder.save_pretrained(os.path.join(prediction_dir, "ctx_encoder_model"))
        scorer.question_encoder.save_pretrained(os.path.join(prediction_dir, "question_encoder_model"))



            # for instance_i, question in enumerate(questions):
            #     # question, context = instance['question'], instance["context"]
            #     pos_context = contexts[instance_i]
            #     pos_score = scorer.score(question, pos_context)

                # _batch_scores.append(score)







