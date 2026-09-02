import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):

    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        output_dim=1024
    ):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_name
        )

        self.bert = BertModel.from_pretrained(
            bert_model_name
        )

        # BERT is used as a frozen language feature extractor.
        # This dramatically reduces CPU training time.
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(
            768,
            output_dim
        )

    @torch.no_grad()
    def encode_bert(self, text_list):
        device = next(self.parameters()).device

        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        outputs = self.bert(**tokens)

        return outputs.last_hidden_state[:, 0, :]

    def forward(self, text_list):

        cls_embedding = self.encode_bert(
            text_list
        )

        return self.projection(
            cls_embedding
        )