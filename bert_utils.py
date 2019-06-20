import collections
import logging
import os
from itertools import chain

import numpy as np
from typing import List, Optional, Set, Iterable

import torch
from spacy.tokens import Span
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import Dataset

from pytorch_pretrained_bert import cached_path
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import load_vocab, WordpieceTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP, \
    VOCAB_NAME, PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP

from torch import nn

from utils import SeqLabelingInstance

logger = logging.getLogger(__name__)

NEG = -1e5

NUM_CONCEPTS = 270

class TokenClassificationInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, doc_id, input_ids, input_mask, segment_ids, label_ids, concept_begin_mask, concept_ids):
        self.doc_id = doc_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.token_begin_mask = concept_begin_mask
        self.concept_ids = concept_ids


def convert_token_classification_examples_to_features(examples: List[SeqLabelingInstance], label_list, max_seq_length, tokenizer):

    features = []
    for (ex_index, example) in enumerate(chain(*examples)):

        input_ids = (tokenizer.convert_tokens_to_ids(["[CLS]"])
                     + example.document_token_ids
                     + tokenizer.convert_tokens_to_ids(["[SEP]"])
                     + example.concepts_token_ids
                     + tokenizer.convert_tokens_to_ids(["[SEP]"]))
        #                   [CLS]        document_tokens                  [SEP]             concept_tokens             [SEP]
        concept_begin_mask = [0] + [0] * len(example.document_token_ids) + [0] + [1] * len(
            example.concepts_token_ids) + [0]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        #            [CLS]        document_tokens                  [SEP]             concept_tokens             [SEP]
        segment_ids = [0] + [0] * len(example.document_token_ids) + [0] + [1] * len(example.concepts_token_ids) + [1]

        #          [CLS]        document_tokens                  [SEP]   concept_labels    [SEP]
        label_ids = [0] + [0] * len(example.document_token_ids) + [0] + example.label_ids.tolist() + [1]

        concept_ids = [0] + [0] * len(example.document_token_ids) + [0] + example.concepts_ids.tolist() + [0]

        assert max_seq_length >= len(input_ids), "The input sequence is too long!"

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_ids += padding
        concept_begin_mask += padding
        concept_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(concept_begin_mask) == max_seq_length
        assert len(concept_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokenizer.convert_ids_to_tokens(input_ids)]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % (" ".join(example.labels)))

        features.append(
            TokenClassificationInputFeatures(doc_id=example.document_id,
                                             input_ids=input_ids,
                                             input_mask=input_mask,
                                             segment_ids=segment_ids,
                                             label_ids=label_ids,
                                             concept_begin_mask=concept_begin_mask,
                                             concept_ids=concept_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def sentences_to_token_classification_examples(sentences):
    examples = []
    for i, sentence in enumerate(sentences):
        tokens = [tok.text for tok in sentence]
        labels = [tok._.label for tok in sentence]
        examples.append(TokenClassificationExample(i, text_a=tokens, labels=labels))

    return examples


def sentences_to_existential_classification_examples(sentences: Iterable[Span]):
    examples = []
    doc_tokens = collections.defaultdict(list)
    doc_labels = {}

    for i, sentence in enumerate(sentences):
        tokens = [tok.text for tok in sentence]
        doc = sentence.doc
        doc_tokens[doc._.pmid].extend(tokens)
        doc_labels[doc._.pmid] = doc._.existential_annotations

    for i, doc in enumerate(doc_tokens):
        examples.append(ExistentialClassificationExample(i, text=doc_tokens[doc], labels=doc_labels[doc]))

    return examples


class BertLabelTokenizer:
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, max_len=None, do_basic_tokenize=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BertTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_basic_tokenize: Whether to do basic tokenization before wordpiece.
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        token_begin_mask = []
        for token in text:
            wordpieces = self.wordpiece_tokenizer.tokenize(token)
            if len(wordpieces) > 0:
                for sub_token in wordpieces:
                    split_tokens.append(sub_token)
                token_begin_mask += [1] + [0] * (len(wordpieces) - 1)
        return split_tokens, token_begin_mask

    def tokenize_labels(self, text, labels):
        split_tokens = []
        split_labels = []
        token_begin_mask = []
        for token, label in zip(text, labels):
            wordpieces = self.wordpiece_tokenizer.tokenize(token)
            if len(wordpieces) > 0:
                for sub_token in wordpieces:
                    split_tokens.append(sub_token)
                split_labels += [label] + ["X"] * (len(wordpieces) - 1)
                token_begin_mask += [1] + [0] * (len(wordpieces) - 1)
        return split_tokens, split_labels, token_begin_mask

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            vocab_file = pretrained_model_name_or_path
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer


class BertForExistentialClassification(BertPreTrainedModel):
    def __init__(self, config, bert, num_labels):
        super(BertForExistentialClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = bert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        out, _ = self.bert(input_ids, token_type_ids, attention_mask,
                           output_all_encoded_layers=False)[:, 0]
        logits = self.classifier(out)
        # TODO Use ranking loss here
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return out_logits


def map_to_orig_tokens(all_labels, all_token_begin_masks):
    # all_labels.shape == (batch_size, length, num_labels)
    mapped_labels = []
    for labels, mask in zip(all_labels, all_token_begin_masks):
        mapped_labels.append(labels[np.where(mask)[0]])

    return mapped_labels


class BertForConceptClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForConceptClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, concept_begin_mask=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if concept_begin_mask is not None:
                active_loss = concept_begin_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, cls_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits


class BertForCLEF19Multilabel(BertPreTrainedModel):
        def __init__(self, config, num_labels):
            super(BertForCLEF19Multilabel, self).__init__(config)
            self.num_labels = num_labels
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.combinator = nn.Linear(config.hidden_size + NUM_CONCEPTS, config.hidden_size + NUM_CONCEPTS)
            self.classifier = nn.Linear(config.hidden_size + NUM_CONCEPTS, num_labels)
            self.apply(self.init_bert_weights)

        def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, concept_sims=None):
            _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            pooled_output = self.dropout(pooled_output)
            combined_features = self.dropout(torch.tanh(torch.cat([pooled_output, concept_sims], dim=1)))
            logits = self.classifier(combined_features)

            if labels is not None:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                return loss
            else:
                return logits
