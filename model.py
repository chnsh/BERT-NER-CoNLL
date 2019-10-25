import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForMaskedLM


class CoNLLClassifier(BertForMaskedLM):
    def __init__(self, config, vocab_size, label_map, disentangled_labels=("B-PER", "I-PER"), dim_size=300):
        super().__init__(config)
        self.label_map = label_map
        self.disentangled_labels = disentangled_labels
        self.embedding = nn.Embedding(vocab_size, dim_size)

        self.context_mlp = nn.Linear(config.hidden_size, config.num_labels)

        self.token_mlp = nn.Linear(config.dim_size, config.num_labels)

        self.token_and_context_mlp = nn.Linear(config.dim_size + config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, label_masks=None, masked_input_ids=None,
                masked_lm_labels=None):
        outputs = self.bert(masked_input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]  # (b, MAX_LEN, 768)

        bert_sequence_reprs = [embedding[mask] for mask, embedding in zip(label_masks, sequence_output)]
        bert_sequence_reprs = pad_sequence(sequences=bert_sequence_reprs, batch_first=True,
                                           padding_value=-1)  # (b, local_max_len, 768)
        sequence_output = self.dropout(bert_sequence_reprs)

        context_logits = self.context_mlp(sequence_output)  # (b, local_max_len, num_labels)

        embeddings = self.embedding(input_ids)  # (b, dim_size)

        token_logits = self.token_mlp(embeddings)

        token_and_context = torch.cat((bert_sequence_reprs, embeddings), dim=-1)

        token_and_context_logits = self.token_and_context_mlp(token_and_context)

        # outputs = (logits,)
        # if labels is not None:
        #     labels = [label[mask] for mask, label in zip(label_masks, labels)]
        #     labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # (b, local_max_len)
        #     loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
        #     mask = labels != -1
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     loss /= mask.float().sum()
        #     outputs = (loss,) + outputs + (labels,)
        #
        # return outputs  # (loss), scores, (hidden_states), (attentions)
