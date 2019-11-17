import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForMaskedLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CoNLLClassifier(BertForMaskedLM):
    def __init__(self, config, embedding_vocab_size, label_map,
                 disentangled_labels=("B-PER", "I-PER"), dim_size=300):
        super().__init__(config)
        self.config = config
        self.label_map = label_map
        self.disentangled_labels = disentangled_labels
        self.embedding = nn.Embedding(embedding_vocab_size, dim_size)

        self.context_mlp = nn.Linear(config.hidden_size, config.num_labels)

        self.token_mlp = nn.Linear(dim_size, config.num_labels)

        self.token_and_context_mlp = nn.Linear(dim_size + config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.linear_combination_weights = nn.Parameter(torch.Tensor(2, 12))
        nn.init.kaiming_uniform_(self.linear_combination_weights, a=math.sqrt(5))

        self.init_weights()

    @property
    def coefficient_weights(self):
        return F.softmax(self.linear_combination_weights, dim=0)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, label_masks=None,
                masked_input_ids=None,
                masked_lm_labels=None):
        outputs = self.bert(masked_input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        is_masked = masked_lm_labels != -1

        sequence_output = outputs[0]  # (b, MAX_LEN, 768)

        bert_sequence_reprs = [embedding[mask] for mask, embedding in
                               zip(label_masks, sequence_output)]

        bert_sequence_reprs = pad_sequence(sequences=bert_sequence_reprs, batch_first=True,
                                           padding_value=-1).to(device)  # (b, local_max_len, 768)

        sequence_output = self.dropout(bert_sequence_reprs)

        context_logits = self.context_mlp(sequence_output)  # (b, local_max_len, num_labels)

        embeddings = [embedding[mask] for mask, embedding in
                      zip(label_masks, self.embedding(input_ids))]  # (b, local_max_len, dim_size)

        embeddings = pad_sequence(sequences=embeddings, batch_first=True,
                                  padding_value=-1)

        token_logits = self.token_mlp(embeddings)  # (b, MAX_LEN, num_labels)

        b, local_max_len, num_labels = token_logits.size()

        stacked_tensors = torch.stack((context_logits, token_logits))

        # broadcast multiply coefficients and stacked tensors to create logits and then sum across
        # stack dimension

        logits = stacked_tensors.view(2, 12, -1) * self.coefficient_weights.unsqueeze(-1)
        logits = logits.sum(dim=0)
        logits = logits.view(b, local_max_len, num_labels)

        outputs = (logits,)
        if labels is not None:
            labels = [label[mask] for mask, label in zip(label_masks, labels)]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = labels != -1
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            loss /= mask.float().sum()
            outputs = (loss,) + outputs + (labels,)

        return outputs  # (loss), scores, (hidden_states), (attentions)
