import torch
from torch import nn
from typing import List
from transformers import (
    AutoTokenizer,
    # PegasusForConditionalGeneration
    AutoModelForSeq2SeqLM
)


class Summarizer(nn.Module):
    def __init__(self, model_name="google/pegasus-xsum", max_length=300):
        super(Summarizer, self).__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def forward(self, src_text: List[str]):

        batch = self.tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(self.device)
        translated = self.model.generate(**batch)
        tgt_text = self.tokenizer.batch_decode(translated, max_length=self.max_length, skip_special_tokens=True)

        if len(src_text) == 1:
            return tgt_text[0]
        else:
            return tgt_text

