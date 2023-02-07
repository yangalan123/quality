import torch
import transformers
from torch.nn import Module

class DPRScorer(Module):
    def __init__(self,
                 context_encoder_name="facebook/dpr-ctx_encoder-multiset-base",
                 question_encoder_name="facebook/dpr-question_encoder-multiset-base",
                 device=None,
                 use_cache=False, verbose=True):
        super(DPRScorer, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_name)
        self.context_encoder = transformers.DPRContextEncoder.from_pretrained(context_encoder_name).to(device)
        self.question_encoder = transformers.DPRQuestionEncoder.from_pretrained(question_encoder_name).to(device)
        self.device = device
        self.use_cache = use_cache
        if use_cache:
            self.cache = {}
        else:
            self.cache = None
        self.verbose = verbose
        self.unk_set = set()

    def _convert_to_batch(self, string):
        # input = self.tokenizer(string, padding="max_length", truncation="only_first")
        # return {
        #     "input_ids": input['input_ids'].to(self.device),
        #     "attention_mask":
        # }

        return {k: torch.tensor([v]).to(self.device) for k, v in self.tokenizer(string).items()}

    def _embed_context(self, context: str):
        context_batch = self._convert_to_batch(context)
        with torch.no_grad():
            out = self.context_encoder(**context_batch)
        return out.pooler_output[0]

    def _embed_question(self, question: str):
        query_batch = self._convert_to_batch(question)
        with torch.no_grad():
            out = self.question_encoder(**query_batch)
        return out.pooler_output[0]

    def embed(self, strings, flag='question'):
        ret = []
        for string in strings:
            _batch = self._convert_to_batch(string)
            if flag == "question":
                enc = self.question_encoder
            else:
                enc = self.context_encoder
            out = enc(**_batch).pooler_output[0]
            ret.append(out)
        return torch.stack(ret)


    def score(self, reference: str, target: str):
        # Reference <- question
        # Target <- context
        if self.use_cache:
            if reference not in self.cache:
                self.cache[reference] = self._embed_question(reference)
            if target not in self.cache:
                self.cache[target] = self._embed_context(target)
            ref_embed = self.cache[reference]
            tgt_embed = self.cache[target]
        else:
            ref_embed = self._embed_question(reference)
            tgt_embed = self._embed_context(target)
        return -torch.norm(ref_embed - tgt_embed)


