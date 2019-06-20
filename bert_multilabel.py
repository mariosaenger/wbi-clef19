from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import random
import numpy as np

import torch
from pandas import DataFrame

from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from bert_utils import BertForMultiLabelSequenceClassification
from utils import DataHandler, EvaluationUtil, LogUtil

# Taken and adapted from
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py

logger = LogUtil.get_logger("bert_multilabel", "_logs")
#

def accuracy_thresh(y_pred: Tensor, y_true: Tensor, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()

class InputInstance(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):

    def __init__(self, train_df: DataFrame, dev_df: DataFrame, test_df: DataFrame):
        self.train_df = train_df
        self.dev_df = dev_df
        self.test_df = test_df

        self.labels = self._get_all_labels(train_df) + self._get_all_labels(dev_df)

    def get_train_instances(self):
        return self._build_instances(self.train_df)

    def get_dev_instances(self):
        return self._build_instances(self.dev_df)

    def get_test_instances(self):
        return self._build_instances(self.test_df)

    def get_labels(self):
        return self.labels

    def _get_all_labels(self, data: DataFrame):
        data = data[data["all_labels"].notna()]
        all_labels = [row["all_labels"].split("|") for _, row in data.iterrows()]
        return list([label for list in all_labels for label in list])

    def _build_instances(self, data: DataFrame):
        instances = list()
        for i, row in data.iterrows():
            gold_labels = row["all_labels"]
            if gold_labels is not None and type(gold_labels) == str:
                gold_labels = gold_labels.split("|")
            else:
                gold_labels = []

            instances.append(InputInstance(guid=i, text_a=row["text"], text_b=None, labels=gold_labels))

        return instances

def convert_examples_to_features(examples, label_encoder, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_ids = []
        for label in example.labels:
            label_ids.append(label_encoder.transform([label])[0])

        label_vector = np.zeros(len(label_encoder.classes_), dtype=float)
        label_vector[label_ids] = 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, ",".join([str(x) for x in label_vector])))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                                      segment_ids=segment_ids, label_ids=label_vector))
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

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_set", choices=DataHandler.ALL_DATA_SET_IDS,required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--additional_data_set", choices=DataHandler.ALL_DATA_SET_IDS, required=False,
                        help="Additional data set to extend the basic training data set. "
                             "Only training data will be used! No additional evaluation data!")

        ## Other parameters
    parser.add_argument("--cache_dir",
                        default="_cache",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=300,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()


    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    data_handler = DataHandler()
    train_data, dev_data = data_handler.get_data_set_by_id(args.data_set)

    if args.additional_data_set is not None:
        logger.info(f"Extending training data with instances from {args.additional_data_set}")
        add_train_data, add_dev_data = data_handler.get_data_set_by_id(args.additional_data_set)

        train_data = train_data.append(add_train_data)
        train_data = train_data.append(add_dev_data)

    logger.info(f"Data set contains {len(train_data)} training and {len(dev_data)} development instances")

    test_data = None
    if args.do_test:
        test_data = data_handler.get_test_data()

    processor = DataProcessor(train_data, dev_data, test_data)
    label_list = processor.get_labels()
    logger.info(f"Labels: {str(label_list)}")

    label_encoder = LabelEncoder()
    label_encoder.fit(label_list)

    num_labels = len(label_encoder.classes_)
    logger.info(f"Num labels: {num_labels}")

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_instances()

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)

    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError \
                ("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError \
                ("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,lr=args.learning_rate,
                              bias_correction=False, max_grad_norm=1.0)

        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                             warmup=args.warmup_proportion, t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    global_batch_no = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(train_examples, label_encoder, args.max_seq_length, tokenizer)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        max_f1 = 0.0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()


                loss_value = loss.item()
                tr_loss += loss_value

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                global_batch_no += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step /num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if (epoch + 1) % 2 == 0:
                eval_examples = processor.get_dev_instances()
                eval_features = convert_examples_to_features(eval_examples, label_encoder, args.max_seq_length, tokenizer)

                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                y_dev = None
                y_dev_pred = None

                y_dev_sigmoid = None

                for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                        logits = model(input_ids, segment_ids, input_mask)

                    # logits = logits.detach().cpu().numpy()
                    # prediction = np.argmax(logits, axis=1)

                    logits_cpu = logits.detach().cpu()
                    logits_sigmoid = logits_cpu.sigmoid()
                    if y_dev_pred is None:
                        y_dev_sigmoid = logits_sigmoid
                    else:
                        y_dev_sigmoid = np.concatenate((y_dev_sigmoid, logits_sigmoid), axis=0)

                    pred_logits = logits.detach().cpu().numpy()
                    if y_dev_pred is None:
                        y_dev_pred = pred_logits
                    else:
                        y_dev_pred = np.concatenate((y_dev_pred, pred_logits), axis=0)

                    if y_dev is None:
                        y_dev = label_ids.detach().cpu().numpy()
                    else:
                        y_dev = np.concatenate((y_dev, label_ids.detach().cpu().numpy()), axis=0)

                    tmp_eval_accuracy = accuracy_thresh(logits, label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples
                loss = tr_loss / nb_tr_steps if args.do_train else None

                i = 0
                gold_labels = dict()
                pred_labels = dict()

                prediction_output = dict()
                for example, true_logits, pred_logits in zip(eval_examples, y_dev, y_dev_sigmoid):
                    true_indexes = np.argwhere(true_logits > 0.0)
                    labels = [label_encoder.inverse_transform(y)[0] for y in true_indexes]

                    pred_indexes = np.argwhere(pred_logits > 0.5)
                    pred = [label_encoder.inverse_transform(y)[0] for y in pred_indexes]

                    gold_labels[str(example.guid)] = labels
                    pred_labels[str(example.guid)] = pred

                    class_logits = {label_encoder.inverse_transform([j])[0]: float(pred_logits[j])
                                    for j in range(len(label_encoder.classes_))}
                    prediction_output[example.guid] = class_logits

                    i += 1

                pred_output_file = os.path.join(args.output_dir, f"prediction_output_{epoch+1}.json")
                json.dump(prediction_output, open(pred_output_file, 'w'), sort_keys=True, indent=2)

                result = {
                    'eval_loss':     eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'global_step':   global_step,
                    'loss':          loss,
                }

                eval_util = EvaluationUtil()

                pred_file = os.path.join(args.output_dir, f"dev_pred_{epoch+1}.txt")
                eval_util.save_predictions(pred_labels, pred_file)

                clef19_result = eval_util.evaluate(pred_labels, gold_labels)

                f1_score = clef19_result["eval_fscore"]
                if f1_score > max_f1:
                    # Save a trained model and the associated configuration
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())

                    max_f1 = f1_score

                icd10_ontology = data_handler.read_icd10_ontology()
                pred_labels_extended = eval_util.extend_paths(pred_labels, icd10_ontology)

                pred_extended_file = os.path.join(args.output_dir, f"dev_pred_extended_{epoch+1}.txt")
                eval_util.save_predictions(pred_labels_extended, pred_extended_file)

                extended_clef19_result = eval_util.evaluate(pred_labels_extended, gold_labels)

                output_eval_file = os.path.join(args.output_dir, f"eval_results_{epoch+1}.txt")
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))

                    clef10_result_str = eval_util.format_result(clef19_result)
                    logger.info(f"CLEF19 evaluation: {clef10_result_str}")
                    writer.write("\nResults prediction:\n")
                    writer.write(clef10_result_str)

                    extended_clef10_result_str = eval_util.format_result(extended_clef19_result)
                    logger.info(f"CLEF19 evaluation (extended): {extended_clef10_result_str}")
                    writer.write("\n\nResults extended prediction:\n")
                    writer.write(extended_clef10_result_str)


    #if not args.do_train:
    # (Re-) Load best model
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    config = BertConfig(output_config_file)
    model = BertForMultiLabelSequenceClassification(config, num_labels=num_labels)
    model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_instances()
        eval_features = convert_examples_to_features(eval_examples, label_encoder, args.max_seq_length, tokenizer)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        y_dev = None
        y_dev_pred = None

        y_dev_sigmoid = None

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            #logits = logits.detach().cpu().numpy()
            #prediction = np.argmax(logits, axis=1)

            logits_cpu = logits.detach().cpu()
            logits_sigmoid = logits_cpu.sigmoid()

            if y_dev_sigmoid is None:
                y_dev_sigmoid = logits_sigmoid
            else:
                y_dev_sigmoid = np.concatenate((y_dev_sigmoid, logits_sigmoid), axis=0)

            pred_logits = logits.detach().cpu().numpy()
            if y_dev_pred is None:
                y_dev_pred = pred_logits
            else:
                y_dev_pred = np.concatenate((y_dev_pred, pred_logits), axis=0)

            if y_dev is None:
                y_dev = label_ids.detach().cpu().numpy()
            else:
                y_dev = np.concatenate((y_dev, label_ids.detach().cpu().numpy()), axis=0)

            tmp_eval_accuracy = accuracy_thresh(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss /nb_tr_steps if args.do_train else None

        i = 0
        gold_labels = dict()
        pred_labels = dict()

        prediction_output = dict()

        for example, true_logits, pred_logits in zip(eval_examples, y_dev, y_dev_sigmoid):
            true_indexes = np.argwhere(true_logits > 0.0)
            labels = [label_encoder.inverse_transform(y)[0] for y in true_indexes]

            pred_indexes = np.argwhere(pred_logits > 0.5)
            pred = [label_encoder.inverse_transform(y)[0] for y in pred_indexes]

            if i < 2:
                logger.info(f"Example: {example.guid}")
                logger.info(f"Example labels: {example.labels}")
                logger.info(f"True labels: {labels}")
                logger.info(f"Pred labels: {pred}")

            gold_labels[str(example.guid)] = labels
            pred_labels[str(example.guid)] = pred

            class_logits = {label_encoder.inverse_transform([j])[0]: float(pred_logits[j])
                             for j in range(len(label_encoder.classes_))}
            prediction_output[example.guid] = class_logits

            i += 1

        pred_output_file = os.path.join(args.output_dir, "prediction_output.json")
        json.dump(prediction_output, open(pred_output_file, 'w'), sort_keys=True, indent=2)

        result = {
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'global_step': global_step,
            'loss': loss,
        }

        eval_util = EvaluationUtil()

        pred_file = os.path.join(args.output_dir, "dev_pred.txt")
        eval_util.save_predictions(pred_labels, pred_file)

        clef19_result = eval_util.evaluate(pred_labels, gold_labels)

        icd10_ontology = data_handler.read_icd10_ontology()
        pred_labels_extended = eval_util.extend_paths(pred_labels, icd10_ontology)

        pred_extended_file = os.path.join(args.output_dir, "dev_pred_extended.txt")
        eval_util.save_predictions(pred_labels_extended, pred_extended_file)

        extended_clef19_result = eval_util.evaluate(pred_labels_extended, gold_labels)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

            clef10_result_str = eval_util.format_result(clef19_result)
            logger.info(f"CLEF19 evaluation: {clef10_result_str}")
            writer.write("\nResults prediction:\n")
            writer.write(clef10_result_str)

            extended_clef10_result_str = eval_util.format_result(extended_clef19_result)
            logger.info(f"CLEF19 evaluation (extended): {extended_clef10_result_str}")
            writer.write("\n\nResults extended prediction:\n")
            writer.write(extended_clef10_result_str)

    if args.do_test:
        test_examples = processor.get_test_instances()
        test_features = convert_examples_to_features(test_examples, label_encoder, args.max_seq_length, tokenizer)

        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        y_dev_sigmoid = None

        for input_ids, input_mask, segment_ids in tqdm(test_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            logits_sigmoid = logits.detach().cpu().sigmoid()
            if y_dev_sigmoid is None:
                y_dev_sigmoid = logits_sigmoid
            else:
                y_dev_sigmoid = np.concatenate((y_dev_sigmoid, logits_sigmoid), axis=0)

        i = 0
        test_labels = dict()
        test_output = dict()

        for example, pred_logits in zip(test_examples, y_dev_sigmoid):
            pred_indexes = np.argwhere(pred_logits > 0.5)
            pred = [label_encoder.inverse_transform(y)[0] for y in pred_indexes]

            if i < 2:
                logger.info(f"Example: {example.guid}")
                logger.info(f"Pred labels: {pred}")

            test_labels[str(example.guid)] = pred

            class_logits = {label_encoder.inverse_transform([j])[0]: float(pred_logits[j])
                            for j in range(len(label_encoder.classes_))}
            test_output[example.guid] = class_logits

            i += 1

        pred_output_file = os.path.join(args.output_dir, "test_output.json")
        json.dump(test_output, open(pred_output_file, 'w'), sort_keys=True, indent=2)

        eval_util = EvaluationUtil()

        test_file = os.path.join(args.output_dir, "test_pred.txt")
        eval_util.save_predictions(test_labels, test_file)

        icd10_ontology = data_handler.read_icd10_ontology()
        test_labels_extended = eval_util.extend_paths(test_labels, icd10_ontology)

        test_extended_file = os.path.join(args.output_dir, "test_pred_extended.txt")
        eval_util.save_predictions(test_labels_extended, test_extended_file)


if __name__ == "__main__":
    main()
