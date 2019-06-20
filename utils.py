import datetime
import logging
import os
from logging import FileHandler

import pandas as pd

from typing import Dict, Tuple, List, Iterable, Union
from pandas import DataFrame
from pytorch_pretrained_bert import BertTokenizer
from random import shuffle, sample
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)


class DataHandler(object):

    LEVEL_2_FT = "level2_ft"
    LEVEL_2_12 = "level2_12"

    LEAF_NODE_FT = "leaf_node_ft"
    LEAF_NODE_12 = "leaf_node_12"

    DRKS_FULL_LEVEL_2= "drks_full_level2"
    DRKS_FULL_LEAF = "drks_full_leaf"

    DRKS_DE_LEAF = "drks_de_leaf"
    DRKS_EN_LEAF = "drks_en_leaf"

    DRKS_DE_LEVEL2 = "drks_de_level2"
    DRKS_EN_LEVEL2 = "drks_en_level2"

    ALL_DATA_SET_IDS = [LEVEL_2_12, LEVEL_2_FT, LEAF_NODE_12, LEAF_NODE_FT,
                        DRKS_FULL_LEVEL_2, DRKS_FULL_LEAF,
                        DRKS_DE_LEAF, DRKS_EN_LEAF,
                        DRKS_DE_LEVEL2, DRKS_EN_LEVEL2]

    LEVEL_2_CLASSIFICATION_FULL_TEXT_DATA_DIR = "data/cl-level2-full"
    LEVEL_2_CLASSIFICATION_LINE12_DATA_DIR = "data/cl-level2-line12"

    MAX_DEPTH_CLASSIFICATION_FULL_TEXT_DATA_DIR = "data/cl-max-depth-full"
    MAX_DEPTH_CLASSIFICATION_LINE12_DATA_DIR = "data/cl-max-depth-line12"

    DRKS_FULL_LEVEL_2_DATA_DIR = "data/cl-drks-level2"
    DRKS_FULL_LEAF_DATA_DIR = "data/cl-drks-leaf-node"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_data_set_by_id(self, data_set_id: str) -> Tuple[DataFrame, DataFrame]:
        if data_set_id not in self.ALL_DATA_SET_IDS:
            raise Exception(f"Unknown data set id {data_set_id}")

        if data_set_id == self.LEVEL_2_12:
            return self.get_level2_classification_data_line12()

        elif data_set_id == self.LEVEL_2_FT:
            return self.get_level2_classification_data_full_text()

        elif data_set_id == self.LEAF_NODE_12:
            return self.get_max_depth_classification_data_line12()

        elif data_set_id == self.LEAF_NODE_FT:
            return self.get_max_depth_classification_data_full_text()

        elif data_set_id == self.DRKS_FULL_LEVEL_2:
            return self.get_drks_full_level2_data()

        elif data_set_id == self.DRKS_DE_LEVEL2:
            return self.get_drks_de_level2_data()

        elif data_set_id == self.DRKS_EN_LEVEL2:
            return self.get_drks_en_level2_data()

        elif data_set_id == self.DRKS_FULL_LEAF:
            return self.get_drks_full_leaf_node_data()

        elif data_set_id == self.DRKS_DE_LEAF:
            return self.get_drks_de_leaf_node_data()

        elif data_set_id == self.DRKS_EN_LEAF:
            return self.get_drks_en_leaf_node_data()

    def get_level2_classification_data_full_text(self) -> Tuple[DataFrame, DataFrame]:
        if not os.path.exists(self.LEVEL_2_CLASSIFICATION_FULL_TEXT_DATA_DIR):
            self.logger.info("Building level 2 classification data set")

            data = self.read_data("data/anns_train_dev.txt", "data/ids_training.txt", "data/ids_development.txt", "data/docs-training")
            icd10_ontology = self.read_icd10_ontology()

            cl_data = self.prepare_leveled_classification_data(data, icd10_ontology, 2, True)
            self.save_classification_data(cl_data, self.LEVEL_2_CLASSIFICATION_FULL_TEXT_DATA_DIR, True)

            self.logger.info("Finished preparation of classification data set")

        else:
            self.logger.info("Classification data set already exists!")

        return self.load_classification_data(self.LEVEL_2_CLASSIFICATION_FULL_TEXT_DATA_DIR)

    def get_level2_classification_data_line12(self) -> Tuple[DataFrame, DataFrame]:
        if not os.path.exists(self.LEVEL_2_CLASSIFICATION_LINE12_DATA_DIR):
            self.logger.info("Building level 2 classification data set")

            data = self.read_data("data/anns_train_dev.txt", "data/ids_training.txt", "data/ids_development.txt", "data/docs-training", [1,2])
            icd10_ontology = self.read_icd10_ontology()

            cl_data = self.prepare_leveled_classification_data(data, icd10_ontology, 2, True)
            self.save_classification_data(cl_data, self.LEVEL_2_CLASSIFICATION_LINE12_DATA_DIR, True)

            self.logger.info("Finished preparation of classification data set")

        else:
            self.logger.info("Classification data set already exists!")

        return self.load_classification_data(self.LEVEL_2_CLASSIFICATION_LINE12_DATA_DIR)

    def get_max_depth_classification_data_line12(self) -> Tuple[DataFrame, DataFrame]:
        return self.get_max_depth_classification_data(self.MAX_DEPTH_CLASSIFICATION_LINE12_DATA_DIR, [1,2])

    def get_max_depth_classification_data_full_text(self) -> Tuple[DataFrame, DataFrame]:
        return self.get_max_depth_classification_data(self.MAX_DEPTH_CLASSIFICATION_FULL_TEXT_DATA_DIR)

    def get_max_depth_classification_data(self, data_set_dir: str, lines: Iterable[int] = None) -> Tuple[DataFrame, DataFrame]:
        if not os.path.exists(data_set_dir):
            self.logger.info("Building max depth classification data set")

            data = self.read_data("data/anns_train_dev.txt", "data/ids_training.txt", "data/ids_development.txt", "data/docs-training", lines)
            icd10_ontology = self.read_icd10_ontology()

            cl_data = self.prepare_max_depth_classification_data(data, icd10_ontology, True)
            self.save_classification_data(cl_data, data_set_dir, True)

            self.logger.info("Finished preparation of classification data set")

        else:
            self.logger.info("Classification data set already exists!")

        return self.load_classification_data(data_set_dir)

    def get_drks_full_level2_data(self) -> Tuple[DataFrame, DataFrame]:
        return self.get_drks_level2_data("data/drks/prepared/drks_full.tsv", self.DRKS_FULL_LEVEL_2_DATA_DIR)

    def get_drks_level2_data(self, drks_tsv_file: str, drks_data_dir: str) -> Tuple[DataFrame, DataFrame]:
        if not os.path.exists(drks_data_dir):
            self.logger.info("Building drks classification data set")

            data = pd.read_csv(drks_tsv_file, sep="\t", index_col="id")
            icd10_ontology = self.read_icd10_ontology()

            cl_data = self.prepare_leveled_classification_data(data, icd10_ontology, 2, True)
            self.save_classification_data(cl_data, drks_data_dir, True)

            self.logger.info("Finished preparation of classification data set")

        else:
            self.logger.info("Classification data set already exists!")

        return self.load_classification_data(drks_data_dir)

    def get_drks_full_leaf_node_data(self):
        return self.get_drks_max_depth_data("data/drks/prepared/drks_full.tsv", self.DRKS_FULL_LEAF_DATA_DIR)

    def get_drks_de_level2_data(self):
        train_data, dev_data = self.get_drks_full_level2_data()
        train_data = train_data[train_data["language"] == "de"]
        dev_data = dev_data[dev_data["language"] == "de"]

        return train_data, dev_data

    def get_drks_en_level2_data(self):
        train_data, dev_data = self.get_drks_full_level2_data()
        train_data = train_data[train_data["language"] == "en"]
        dev_data = dev_data[dev_data["language"] == "en"]

        return train_data, dev_data

    def get_drks_de_leaf_node_data(self):
        train_data, dev_data = self.get_drks_full_leaf_node_data()
        train_data = train_data[train_data["language"] == "de"]
        dev_data = dev_data[dev_data["language"] == "de"]

        return train_data, dev_data

    def get_drks_en_leaf_node_data(self):
        train_data, dev_data = self.get_drks_full_leaf_node_data()
        train_data = train_data[train_data["language"] == "en"]
        dev_data = dev_data[dev_data["language"] == "en"]

        return train_data, dev_data

    def get_drks_max_depth_data(self, drks_tsv_file: str, data_set_dir: str) -> Tuple[DataFrame, DataFrame]:
        if not os.path.exists(data_set_dir):
            self.logger.info("Building max depth classification data set")

            data = pd.read_csv(drks_tsv_file, sep="\t", index_col="id")
            icd10_ontology = self.read_icd10_ontology()

            cl_data = self.prepare_max_depth_classification_data(data, icd10_ontology, True)
            self.save_classification_data(cl_data, data_set_dir, True)

            self.logger.info("Finished preparation of classification data set")

        else:
            self.logger.info("Classification data set already exists!")

        return self.load_classification_data(data_set_dir)

    def load_classification_data(self, data_directory: str) -> Tuple[DataFrame, DataFrame]:
        train_file = os.path.join(data_directory, "train.tsv")
        self.logger.info(f"Loading classification training data from {train_file}")

        train_data = self.read_classification_data(train_file)
        self.logger.info(f"Found {len(train_data)} training instances")

        dev_file = os.path.join(data_directory, "dev.tsv")
        self.logger.info(f"Loading classification development data from {dev_file}")

        dev_data = self.read_classification_data(dev_file)
        self.logger.info(f"Found {len(dev_data)} development instances")

        return train_data, dev_data

    def read_data(self, label_file: str, train_ids_file: str, dev_ids_file: str, doc_directory: str, lines: List[int] = None) -> DataFrame:
        # Read train and dev ids
        train_ids = self.read_ids(train_ids_file, "train")
        dev_ids = self.read_ids(dev_ids_file, "dev")
        id_to_data_set = pd.concat([train_ids, dev_ids])

        # Read document texts
        texts = self.read_texts(doc_directory, lines)

        # Read label data
        data = pd.read_csv(label_file, sep="\t", names=["id", "all_labels"], index_col="id", encoding="utf8")

        # Join information
        joined_data = id_to_data_set.join(data)
        joined_data = joined_data.join(texts)

        return joined_data

    def get_test_data(self, test_dir: str = "data/docs-test") -> DataFrame:
        test_df = self.read_texts(test_dir)
        test_df["all_labels"] = None
        test_df["label"] = None

        return test_df

    def prepare_leveled_classification_data(self, data: DataFrame, icd10_ontology: Dict[str, str], level: int, include_none: bool) -> DataFrame:
        id_to_label = dict()
        for i, row in data.iterrows():
            class_label = None

            all_labels = row["all_labels"]
            if all_labels is not None and not type(all_labels) == float:
                for label in all_labels.split("|"):
                    path = icd10_ontology[label]
                    if len(path.split("#")) == level:
                        class_label = label

            # Only use data which have a label at this level!
            if class_label is not None:
                id_to_label[i] = class_label
            elif include_none:
                id_to_label[i] = "NoLabel"

        class_labels = DataFrame.from_dict(id_to_label, orient="index", columns=["label"])
        joined_data = class_labels.join(data)

        return joined_data

    def prepare_max_depth_classification_data(self, data: DataFrame, icd10_ontology: Dict[str, str], include_none: bool) -> DataFrame:
        id_to_label = dict()
        for i, row in data.iterrows():
            max_depth = -1
            max_depth_label = None

            all_labels = row["all_labels"]
            if all_labels is not None and not type(all_labels) == float:

                for label in all_labels.split("|"):
                    path = icd10_ontology[label]
                    path_components = path.split("#")

                    if len(path_components) > max_depth:
                        max_depth = len(path_components)
                        max_depth_label = path_components[-1]

            # Only which have a label at this level!
            if max_depth_label is not None:
                id_to_label[i] = max_depth_label
            elif include_none:
                id_to_label[i] = "NoLabel"

        class_labels = DataFrame.from_dict(id_to_label, orient="index", columns=["label"])
        joined_data = class_labels.join(data)

        return joined_data

    def save_classification_data(self, data: DataFrame, output_dir: str, include_none: bool):
        os.makedirs(output_dir, exist_ok=True)
        data.to_csv(os.path.join(output_dir, "full.tsv"), sep="\t", index_label="id")

        if not include_none:
            data = data[data["label"] != "NoLabel"]

        train_data = data[data["data_set"] == "train"]
        train_data.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index_label="id")

        dev_data = data[data["data_set"] == "dev"]
        dev_data.to_csv(os.path.join(output_dir, "dev.tsv"), sep="\t", index_label="id")

    def read_classification_data(self, input_file):
        return pd.read_csv(input_file, sep="\t", index_col="id")

    def read_icd10_ontology(self, file: str = "icd10syst2016_codePaths.txt") -> Dict[str, str]:
        icd10_to_path = dict()

        with open(file, "r", encoding="utf8") as input:
            for line in input.readlines():
                columns = line.strip().split(";")
                icd10_code = columns[0]
                path = "#".join(reversed(columns[:]))

                icd10_to_path[icd10_code] = path

        return icd10_to_path

    def read_reduced_icd10_ontology(self, ontology_file: str = "icd10syst2016_codePaths.txt") -> Dict[str, str]:
        icd10_ontology = self.read_icd10_ontology(ontology_file)
        icd10_sections = [code for code, path in icd10_ontology.items()
                         if len(path.split("#")) <= 2 or ((len(path.split("#")) <= 4 and "-" in code))]

        reduced_ontology = dict()
        for icd10_code in icd10_sections:
            reduced_ontology[icd10_code] = icd10_ontology[icd10_code]

        return reduced_ontology

    def read_ids(self, ids_file: str, label: str) -> DataFrame:
        ids_data = pd.read_csv(ids_file, names=["id"])
        ids_data = ids_data.set_index(["id"])
        ids_data["data_set"] = label
        return ids_data

    def read_texts(self, directory: str, lines: List[int] = None) -> DataFrame:
        id_to_text = dict()
        for file in os.listdir(directory):
            # There is an id.txt file in doc-training explaining the structure of the files
            if file == "id.txt":
                continue

            id = int(file.replace(".txt", ""))
            with open(os.path.join(directory, file), "r", encoding="utf-8") as reader:
                if lines is None:
                    text = reader.read().strip().replace("\n", " ")
                    id_to_text[id] = text
                else:
                    line_no = 1
                    text = ""
                    for line in reader.readlines():
                        if line_no in lines:
                            text += line.strip() + " "
                        line_no += 1

                    id_to_text[id] = text.strip()

        data = DataFrame.from_dict(id_to_text, orient="index", columns=["text"])
        return data

    def read_all_icd10_concept_names(self, file: str = "preAnalysis/icd10systalpha2016_concepts.txt") -> Dict[str, str]:
        icd10_to_concept = dict()

        with open(file, "r") as reader:
            for line in reader.readlines():
                columns = line.strip().split("\t")
                if len(columns) == 2:
                    concept_name = columns[1]
                    # if concept_name.endswith("."):
                    #     self.logger.warning(f"Removing trailing . from {concept_name}")
                    #     concept_name = concept_name[:-1]

                    icd10_to_concept[columns[0]] = concept_name
                else:
                    self.logger.warning(f"Found no concept name in line {line.strip()}")
            reader.close()

        return icd10_to_concept

    def read_icd10_section_concept_names(self, ontology_file: str = "icd10syst2016_codePaths.txt",
                                         concept_name_file: str = "icd10systalpha2016_concepts.txt") -> Dict[str, str]:
        reduced_icd10_ontology = self.read_reduced_icd10_ontology(ontology_file)
        all_concept_names = self.read_all_icd10_concept_names(concept_name_file)

        icd10_section_names = dict()
        for code in reduced_icd10_ontology.keys():
            icd10_section_names[code] = all_concept_names[code]

        return icd10_section_names


class SeqLabelingInstance(object):
    """A class used to represent a single training instance for the sequence labeling setup.

        Attributes:
            document_id: The unique identifier of the document
            document_labels: List of strings representing the goldstandard labels (e.g. C70-C85) or None, if test instance
            document_tokens: List of strings representing the word piece tokens of the document (left side)
            document_token_ids: List of integers representing the word piece token ids of the document (left side)
            concept_tokens: List of strings representing the word piece tokens of the concepts names (right side)
                (The single concepts are separated by an separator sign)
            concept_token_ids: List of integers representing the word piece token ids of the concept names (right side)
                (The single concepts are separated by an separator sign resp. the id of the separator sign)
            concepts: List of strings representing the concepts (e.g. C70-C85) at the particular position (at the right side)
            concepts_ids: List of integers representing the concept ids at the particular position (at the right side)
            labels: List of strings representing the target labels for the concept names (right side) or None, if testing instance
            label_ids: List of integers represeting the target label ids for the concept names (right side) or None, if testing instance
            is_positive_sample: Indicates whether this instance contains a true positive concept at the right side
    """

    def __init__(self, doc_id: str, doc_labels: Iterable[str], doc_tokens: Iterable[str], doc_token_ids: Iterable[int],
                 concept_tokens: Iterable[str], concept_token_ids: Iterable[int], concepts: Iterable[str], concept_ids: Iterable[int],
                 labels: Iterable[str], label_ids: Iterable[int], is_positive_sample: bool):
        self.document_id = doc_id
        self.document_labels = doc_labels

        self.document_tokens = doc_tokens
        self.document_token_ids = doc_token_ids

        self.concepts_tokens = concept_tokens
        self.concepts_token_ids = concept_token_ids
        self.concepts = concepts
        self.concepts_ids = concept_ids

        self.labels = labels
        self.label_ids = label_ids

        self.is_positive_sample = is_positive_sample

    def __str__(self):
        return f"ID              : {self.document_id}\n" \
            f"Document-Labels    : {self.document_labels}\n" \
            f"Document-Tokens    : {self.document_tokens}\n" \
            f"Document-Token-IDs : {self.document_token_ids}\n" \
            f"Concepts-Tokens    : {self.concepts_tokens}\n" \
            f"Concepts-Token-IDs : {self.concepts_token_ids}\n" \
            f"Concepts           : {self.concepts}\n" \
            f"Concept-IDs        : {self.concepts_ids}\n" \
            f"Labels             : {self.labels}\n" \
            f"Label-IDs          : {self.label_ids}\n" \
            f"Is-Positive        : {self.is_positive_sample}"

    def to_csv_lines(self):
        return f"ID\t{self.document_id}\n" \
            "Document-Labels\t" + "\t".join(self.document_labels) + "\n" \
            "Document-Tokens\t" + "\t".join(self.document_tokens) + "\n" \
            "Document-Token-IDs\t" + "\t".join([str(i) for i in self.document_token_ids]) + "\n" \
            "Concept-Tokens\t" + "\t".join(self.concepts_tokens) + "\n" \
            "Concept-Token-IDs\t" + "\t".join([str(i) for i in self.concepts_token_ids]) + "\n" \
            "Concepts\t" + "\t".join(self.concepts) + "\n" \
            "Concept-IDs\t" + "\t".join([str(i) for i in self.concepts_ids]) + "\n" \
            "Labels\t" + "\t".join(self.labels) + "\n" \
            "Label-IDs\t" + "\t".join([str(i) for i in self.label_ids]) + "\n" \
            f"Is-Positive\t{self.is_positive_sample}\n"


class DataSet(object):
    """ Class to represent a data set. A data set contains a list of sequence labeling instances.
        Furthermore the tokenizer as well as the icd10 and label encoders used to create the
        sequence labeling instances are given.

        Attributes:
            instances: List of lists of sequence labeling instances. Each list holds all the sequence labeling instances
                for one training / test instance.
            tokenizer: BertTokenizer used to tokenize the documents and concept names during instance creation
            icd10_encoder: LabelEncoder used to encode the ICD10 label (e.g. C70-C89) during instance creation
            seq_label_encoder: LabelEncoder used to encode the sequence labels (B, I, X, O) during instance creation
    """

    def __init__(self, instances: Iterable[Iterable[SeqLabelingInstance]], tokenizer: BertTokenizer,
                 icd10_encoder: LabelEncoder, seq_label_encoder: LabelEncoder):
        self.instances = instances
        self.tokenizer = tokenizer
        self.icd10_encoder = icd10_encoder
        self.seq_label_encoder = seq_label_encoder


class SeqLabelingDataGenerator(object):

    def __init__(self, tokenizer: BertTokenizer = None, max_doc_length: int = 220, num_concepts_per_sample: int = 4,
                 max_con_length: int = 25, wordpiece_label: str = "X", con_separator: str = "."):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.max_doc_length = max_doc_length
        self.num_concepts_per_sample = num_concepts_per_sample
        self.max_con_length = max_con_length
        self.wordpiece_label = wordpiece_label
        self.con_separator = con_separator

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
        else:
            self.tokenizer = tokenizer

        self.data_handler = DataHandler()
        self.icd10_concept_names = self.data_handler.read_icd10_section_concept_names()

        self.icd10_encoder = LabelEncoder()
        self.icd10_encoder.fit(sorted(sorted(self.icd10_concept_names.keys()) + ["O"]))

        self.seq_label_encoder = LabelEncoder()
        self.seq_label_encoder.fit(sorted(["B", "I", wordpiece_label, "O"]))

        # Pre-calculate / -tokenize concept names
        self._tokenize_concept_names(self.icd10_concept_names, self.tokenizer)

        # Separator token and token id to distinguish start / end of the different concepts in an instance
        self.separator_token = self.tokenizer.tokenize(self.con_separator)
        self.separator_token_id = self.tokenizer.convert_tokens_to_ids(self.separator_token)

        # Store loaded data sets
        self.data_set_cache = dict()

        # Two-fold mapping: (1) map from data_id to list of documents (2) map from document to word piece token / token ids
        self.data_set_to_document_to_tokens = dict()
        self.data_set_to_document_to_tokens_ids = dict()

    def get_data_set(self, data_set_id: str, additional_data_set: str,
                     shuffle_icd10: bool = True, num_negative_samples: int = None) -> DataSet:
        data_set = None
        if data_set_id in self.data_set_cache:
            self.logger.info(f"Re-using cached data set {data_set_id}")
            data_set = self.data_set_cache[data_set_id]
        else:
            if data_set_id == "TRAIN":
                # TODO: The level 2 data set contains all labels, too! The name is deceptive!
                data_set, _ = self.data_handler.get_level2_classification_data_line12()
            elif data_set_id == "DEV":
                _, data_set = self.data_handler.get_level2_classification_data_line12()
            else:
                raise Exception(f"Unknown data set {data_set_id}")

        if additional_data_set is not None:
            self.logger.info(f"Extending training data with instances from {additional_data_set}")
            add_train_data, add_dev_data = self.data_handler.get_data_set_by_id(additional_data_set)

            data_set = data_set.append(add_train_data)
            data_set = data_set.append(add_dev_data)

        self.logger.info(f"Data set contains {len(data_set)} instances")
        self.data_set_cache[data_set_id] = data_set

        instances = self.generate_training_data(data_set_id, data_set, shuffle_icd10, num_negative_samples)

        return DataSet(instances, self.tokenizer, self.icd10_encoder, self.seq_label_encoder)

    def generate_training_data(self, data_set_id: str, data: DataFrame, shuffle_icd10: bool, num_negative_samples: int) -> Iterable[Iterable[SeqLabelingInstance]]:
        # Pre-calculate document tokenization
        self._tokenize_document_text(data_set_id, data, self.tokenizer)

        document_to_tokens = self.data_set_to_document_to_tokens[data_set_id]
        document_to_token_ids = self.data_set_to_document_to_tokens_ids[data_set_id]

        all_training_samples = []

        self.logger.info("Generating sequence labeling instances")
        icd10_items = list(self.icd10_concept_names.items())
        for i, row in tqdm(data.iterrows(), total=len(data), desc="generate_instances"):
            doc_tokens = document_to_tokens[i]
            doc_token_ids = document_to_token_ids[i]

            doc_labels = str(row["all_labels"]).split("|")

            # List of all training instances for this particular document!
            training_instances = []

            if shuffle_icd10:
                # Shuffle icd10 items to get unique order
                shuffle(icd10_items)
            else:
                icd10_items = sorted(icd10_items)

            # Iterate over all ICD-10 with the given (ICD10) group size
            for position in range(0, len(icd10_items), self.num_concepts_per_sample):
                group = icd10_items[position:position + self.num_concepts_per_sample]

                iob_labels = []
                concept_labels = []
                full_con_tokens = []
                full_con_token_ids = []

                # Indicates whether this group of ICD10 codes contains at least
                # one code of the current document!
                contains_true_positive = False

                for code, _ in group:
                    is_true_positive = code in doc_labels
                    con_tokens = self.icd10_to_tokens[code][:self.max_con_length]
                    con_token_ids = self.icd10_to_token_ids[code][:self.max_con_length]

                    full_con_tokens += con_tokens + self.separator_token
                    full_con_token_ids += con_token_ids + self.separator_token_id

                    concept_labels += [code]*len(con_tokens) + ["O"]

                    if is_true_positive:
                        # Every concept should start with a normal word. Subsequent tokens may are word pieces
                        # Word pieces will be label with <wordpiece_label> (e.g. X)
                        iob_labels += ["B"] + ["I" if not t.startswith("##") else self.wordpiece_label for t in con_tokens[1:]]
                        contains_true_positive = True
                    else:
                        iob_labels += ["O"] * len(con_tokens)

                    # Append label for concept separator
                    iob_labels += ["O"]

                # Encode IOB- and concept labels
                labels_encoded = self.seq_label_encoder.transform(iob_labels)
                concept_ids = self.icd10_encoder.transform(concept_labels)

                #TODO: Dirty hack
                if type(i) == str and (i.startswith("de_") or i.startswith("en_")):
                    i = i.replace("de_", "").replace("en_", "")

                    if "DRKS" in i:
                        i = int(i.replace("DRKS", "")) + 200000

                doc_id = int(i)

                instance = SeqLabelingInstance(doc_id, doc_labels, doc_tokens, doc_token_ids, full_con_tokens, full_con_token_ids,
                                               concept_labels, concept_ids, iob_labels, labels_encoded, contains_true_positive)
                training_instances.append(instance)

            if num_negative_samples is not None:
                positive_instances = [i for i in training_instances if i.is_positive_sample]

                negative_instances = [i for i in training_instances if not i.is_positive_sample]
                negative_instances = sample(negative_instances, num_negative_samples)

                training_instances = positive_instances + negative_instances

            all_training_samples += [training_instances]

        return all_training_samples

    def _tokenize_concept_names(self, icd10_concept_names: Dict[str, str], tokenizer: BertTokenizer):
        # Pre-calculate / -tokenize concept names
        self.icd10_to_tokens = dict()
        self.icd10_to_token_ids = dict()
        for icd10_code, concept_text in icd10_concept_names.items():
            tokens = tokenizer.tokenize(concept_text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            self.icd10_to_tokens[icd10_code] = tokens
            self.icd10_to_token_ids[icd10_code] = token_ids

    def _tokenize_document_text(self, data_set_id: str, data: DataFrame, tokenizer: BertTokenizer):
        if data_set_id in self.data_set_to_document_to_tokens:
            self.logger.info(f"Skip document tokenization - already tokenized data set {data_set_id}")
            return

        self.logger.info(f"Tokenizing document texts from data set {data_set_id}")
        document_to_tokens = dict()
        document_to_token_ids = dict()

        for i, row in tqdm(data.iterrows(), total=len(data), desc="tokenize_documents"):
            tokens = tokenizer.tokenize(row["text"])[:self.max_doc_length]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            document_to_tokens[i] = tokens
            document_to_token_ids[i] = token_ids

        self.data_set_to_document_to_tokens[data_set_id] = document_to_tokens
        self.data_set_to_document_to_tokens_ids[data_set_id] = document_to_token_ids

class PathSeqLabelingDataGenerator(object):

    def __init__(self, tokenizer: BertTokenizer = None, max_doc_length: int = 400,  max_con_length: int = 26,
                 wordpiece_label: str = "X", con_separator: str = "."):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.max_doc_length = max_doc_length
        self.max_con_length = max_con_length
        self.wordpiece_label = wordpiece_label
        self.con_separator = con_separator

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
        else:
            self.tokenizer = tokenizer

        self.data_handler = DataHandler()
        self.icd10_concept_names = self.data_handler.read_icd10_section_concept_names()

        self.icd10_ontology = self.data_handler.read_reduced_icd10_ontology()
        self._calculate_pos_and_negative_paths(self.icd10_ontology)

        self.icd10_encoder = LabelEncoder()
        self.icd10_encoder.fit(sorted(sorted(self.icd10_concept_names.keys()) + ["O"]))

        self.seq_label_encoder = LabelEncoder()
        self.seq_label_encoder.fit(sorted(["B", "I", wordpiece_label, "O", "S"]))

        # Pre-calculate / -tokenize concept names
        self._tokenize_concept_names(self.icd10_concept_names)

        # Separator token and token id to distinguish start / end of the different concepts in an instance
        self.separator_token = self.tokenizer.tokenize(self.con_separator)
        self.separator_token_id = self.tokenizer.convert_tokens_to_ids(self.separator_token)

        # Store loaded data sets
        self.data_set_cache = dict()

        # Two-fold mapping: (1) map from data_id to list of documents (2) map from document to word piece token / token ids
        self.data_set_to_document_to_tokens = dict()
        self.data_set_to_document_to_tokens_ids = dict()

    def _calculate_pos_and_negative_paths(self, icd10_ontology: Dict[str, str]):
        self.icd10_to_pos_and_neg_paths = dict()

        for code, path in icd10_ontology.items():
            partial_positive_paths = set()
            negative_paths = set()

            path_parts = set(path.split("#"))

            for _, other_path in icd10_ontology.items():
                if path == other_path:
                    continue

                if len(path_parts.intersection(set(other_path.split("#")))) > 0:
                    partial_positive_paths.add(other_path)
                else:
                    negative_paths.add(other_path)

            self.icd10_to_pos_and_neg_paths[code] = (partial_positive_paths, negative_paths)

    def get_data_set(self, data_set_id: str, additional_data_set: str, is_training_mode: bool,
                     num_extra_positive_samples_per_label: int=3, num_negative_samples: int=15) -> DataSet:
        data_set = None
        if data_set_id in self.data_set_cache:
            self.logger.info(f"Re-using cached data set {data_set_id}")
            data_set = self.data_set_cache[data_set_id]
        else:
            if data_set_id == "TRAIN":
                # TODO: The level 2 data set contains all labels, too! The name is deceptive!
                data_set, _ = self.data_handler.get_level2_classification_data_full_text()
            elif data_set_id == "DEV":
                _, data_set = self.data_handler.get_level2_classification_data_full_text()
            else:
                raise Exception(f"Unknown data set {data_set_id}")

        if additional_data_set is not None:
            self.logger.info(f"Extending training data with instances from {additional_data_set}")
            add_train_data, add_dev_data = self.data_handler.get_data_set_by_id(additional_data_set)

            data_set = data_set.append(add_train_data)
            data_set = data_set.append(add_dev_data)

        self.logger.info(f"Data set contains {len(data_set)} instances")

        self.data_set_cache[data_set_id] = data_set

        instances = self.generate_training_data(data_set_id, data_set, is_training_mode, num_extra_positive_samples_per_label, num_negative_samples)

        return DataSet(instances, self.tokenizer, self.icd10_encoder, self.seq_label_encoder)

    def generate_training_data(self, data_set_id: str, data: DataFrame, is_training_mode: bool, num_extra_positive_samples_per_label: int,
                               num_negative_samples: int) -> Iterable[Iterable[SeqLabelingInstance]]:
        all_seq_labeling_instances = []

        # Pre-calculate document tokenization
        self._tokenize_document_text(data_set_id, data, self.tokenizer)

        self.logger.info("Generating sequence labeling instances")
        for i, row in tqdm(data.iterrows(), total=len(data), desc="generate_instances"):
            # FIXME: This is quite dirty!
            all_labels = str(row["all_labels"])
            if all_labels != "nan":
                document_icd10_codes = all_labels.split("|")
            else:
                document_icd10_codes = []

            seq_labeling_instances = []

            if is_training_mode:

                # Generate positive and negative samples for each ICD10 code label
                for icd10_code in document_icd10_codes:
                    path = self.icd10_ontology[icd10_code]

                    # Generate exact positive instance
                    exact_positive_instance = self._generate_seq_labeling_instance(data_set_id, i, document_icd10_codes, path, path, True)
                    seq_labeling_instances.append(exact_positive_instance)

                    # Get pre-calculated partial positive and negative paths
                    partial_positive_paths, negative_paths = self.icd10_to_pos_and_neg_paths[icd10_code]

                    # Sample <num_positive_samples_per_label> instances for this training document
                    partial_positive_paths = sample(partial_positive_paths, num_extra_positive_samples_per_label)

                    # Generate partial positive instances
                    for partial_pos_path in partial_positive_paths:
                        partial_pos_instance = self._generate_seq_labeling_instance(data_set_id, i, document_icd10_codes, path, partial_pos_path, True)
                        seq_labeling_instances.append(partial_pos_instance)

                    # Sample <num_negative_samples> as negative signals for training
                    negative_paths = sample(negative_paths, num_negative_samples)

                    # Generate negative instances
                    for negative_path in negative_paths:
                        negative_instance = self._generate_seq_labeling_instance(data_set_id, i, document_icd10_codes, path, negative_path, False)
                        seq_labeling_instances.append(negative_instance)

                # Edge case - document has no gold standard labels
                if len(document_icd10_codes) == 0:
                    icd10_items = list(self.icd10_ontology.items())

                    # Sample <num_negative_samples> as negative signals for training
                    icd10_items = sample(icd10_items, num_negative_samples)

                    # Generate negative instances
                    for _, icd10_path in icd10_items:
                        seq_labeling_instance = self._generate_seq_labeling_instance(data_set_id, i, document_icd10_codes, "", icd10_path, False)
                        seq_labeling_instances.append(seq_labeling_instance)

            else:
                for icd10_code, path in self.icd10_ontology.items():
                    is_true_positive = icd10_code in document_icd10_codes

                    seq_labeling_instance = self._generate_seq_labeling_instance(data_set_id, i, document_icd10_codes, "", path, is_true_positive)
                    seq_labeling_instances.append(seq_labeling_instance)

            all_seq_labeling_instances += [seq_labeling_instances]

        return all_seq_labeling_instances

    def _generate_seq_labeling_instance(self, data_set_id: str, document_id: str, document_labels: Iterable[str],
                                        instance_path: str, path_to_encode: str, is_positive_instance: bool):
        document_tokens = self.data_set_to_document_to_tokens[data_set_id][document_id]
        document_token_ids = self.data_set_to_document_to_tokens_ids[data_set_id][document_id]

        iob_labels = []
        concept_labels = []
        full_con_tokens = []
        full_con_token_ids = []

        instance_path_parts = instance_path.split("#")
        to_encode_path_parts = path_to_encode.split("#")

        for path_part in to_encode_path_parts:
            con_tokens = self.icd10_to_tokens[path_part][:self.max_con_length]
            con_token_ids = self.icd10_to_token_ids[path_part][:self.max_con_length]

            full_con_tokens += con_tokens + self.separator_token
            full_con_token_ids += con_token_ids + self.separator_token_id

            concept_labels += [path_part] * len(con_tokens) + ["O"]

            if path_part in instance_path_parts or path_part in document_labels:
                iob_labels += ["B"] if instance_path.startswith(path_part) else ["I"]
                iob_labels += ["I" if not t.startswith("##") else self.wordpiece_label for t in con_tokens[1:]]
            else:
                iob_labels += ["O"] * len(con_tokens)

            # Append label for concept separator
            iob_labels += ["S"]

        # Encode IOB- and concept labels
        labels_encoded = self.seq_label_encoder.transform(iob_labels)
        concept_ids = self.icd10_encoder.transform(concept_labels)

        return SeqLabelingInstance(document_id, document_labels, document_tokens, document_token_ids, full_con_tokens, full_con_token_ids,
                                           concept_labels, concept_ids, iob_labels, labels_encoded, is_positive_instance)

    def _tokenize_concept_names(self, icd10_concept_names: Dict[str, str]):
        # Pre-calculate / -tokenize concept names
        self.icd10_to_tokens = dict()
        self.icd10_to_token_ids = dict()
        for icd10_code, concept_text in icd10_concept_names.items():
            tokens = self.tokenizer.tokenize(concept_text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            self.icd10_to_tokens[icd10_code] = tokens
            self.icd10_to_token_ids[icd10_code] = token_ids

    def _tokenize_document_text(self, data_set_id: str, data: DataFrame, tokenizer: BertTokenizer):
        if data_set_id in self.data_set_to_document_to_tokens:
            self.logger.info(f"Skip document tokenization - already tokenized data set {data_set_id}")
            return

        self.logger.info(f"Tokenizing document texts from data set {data_set_id}")
        document_to_tokens = dict()
        document_to_token_ids = dict()

        for i, row in tqdm(data.iterrows(), total=len(data), desc="tokenize_documents"):
            tokens = tokenizer.tokenize(row["text"])[:self.max_doc_length]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            document_to_tokens[i] = tokens
            document_to_token_ids[i] = token_ids

        self.data_set_to_document_to_tokens[data_set_id] = document_to_tokens
        self.data_set_to_document_to_tokens_ids[data_set_id] = document_to_token_ids


class EvaluationUtil(object):

    def extend_paths(self, predictions: Dict[str, Union[str, Iterable]], icd10_ontology: Dict[str, str],
                     no_label_class: str="NoLabel") -> Dict[str, Iterable[str]]:

        extend_result = dict()
        for doc_id, prediction in predictions.items():
            labels = set()

            pred_labels = []
            if type(prediction) == list:
                pred_labels = prediction
            else:
                pred_labels.append(prediction)

            for label in pred_labels:
                if label != no_label_class:
                    path = icd10_ontology[label]
                    for l in path.split("#"):
                        labels.add(l)

            extend_result[doc_id] = labels

        return extend_result

    def evaluate(self, pred_labels, true_labels):
        tps = 0
        fps = 0
        fns = 0

        for id in true_labels.keys():
            anns_id = true_labels[id]
            preds_id = pred_labels[id]

            for pred in preds_id:
                if pred in anns_id:
                    tps += 1
                    # out_file.write('TP\t' + id + '\t' + pred + '\n')
                else:
                    fps += 1
                    # out_file.write('FP\t' + id + '\t' + pred + '\n')

            for ann in anns_id:
                if ann not in preds_id:
                    fns += 1
                    # out_file.write('FN\t' + id + '\t' + ann + '\n')

        precision = tps / max(tps + fps, 1)
        recall = tps / max(tps + fns, 1)
        fscore = 2 * precision * recall / max(precision + recall, 1)

        result = {
            "eval_true_positives": tps,
            "eval_false_positives": fps,
            "eval_false_negatives": fns,

            "eval_precision": precision,
            "eval_recall": recall,
            "eval_fscore": fscore
        }

        return result

    def format_result(self, result: Dict):
        result_formatted = ""
        for key, value in result.items():
            result_formatted += f"{key}={value}\n"

        return result_formatted

    def save_predictions(self, prediction: Dict[str, Iterable[str]], output_file: str):
        with open(output_file, "w") as output_writer:
            for doc_id, labels in prediction.items():
                if len(labels) > 0:
                    label_str = "|".join(labels)
                    output_writer.write(f"{doc_id}\t{label_str}\n")
                else:
                    output_writer.write(f"{doc_id}\n")

            output_writer.close()

    def build_gold_standard_labels(self, data: DataFrame):
        gold_labels = dict()

        for doc_id, values in data.iterrows():
            labels = values["all_labels"]
            if type(labels) != float:
                gold_labels[doc_id] = set(values["all_labels"].split("|"))
            else:
                gold_labels[doc_id] = set()

        return gold_labels


def export_data_set(data_set : DataSet, output_file: str):

    with open(output_file, "w", encoding="utf-8") as writer:
        for entry in data_set.instances:
            reached_negative = False

            for instance in entry:
                if not reached_negative and not instance.is_positive_sample:
                    writer.write("\n\n")
                    reached_negative = True

                writer.write(instance.to_csv_lines())
                writer.write("\n")

            writer.write("\n\n\n")

        writer.close()

class LogUtil(object):

    @staticmethod
    def get_logger(logger_name: str, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        logger = logging.getLogger(logger_name)

        log_file = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        logger.addHandler(FileHandler(log_file, encoding="utf-8"))

        return logger


if __name__ == "__main__":
    #train_instances = generator.generate_training_data(samples, icd10_concepts, tokenizer, 4, 300, 50)

    handler = DataHandler()

    # data_set, _ = handler.get_level2_classification_data_line12()
    # data_set = data_set[data_set["all_labels"].notna()]
    # print(data_set)
    #
    # generator = PathSeqLabelingDataGenerator()
    #
    # data_set = generator.get_data_set("TRAIN", True, 1, 2)
    # export_data_set(data_set, "_tmp/test1.tsv")
    #
    # data_set = generator.get_data_set("DEV", False)
    # export_data_set(data_set, "_tmp/test2.tsv")
    #
    # print(len(data_set.instances))
    # print(sum([len(i) for i in data_set.instances]))

    # train_data, dev_data = handler.get_data_set_by_id(handler.DRKS_DE_LEAF)
    # print(train_data.head())
    # train_data, dev_data = handler.get_data_set_by_id(handler.DRKS_EN_LEAF)
    # print(train_data.head())

    test_data = handler.get_test_data()
    print(len(test_data))
    print(test_data.head())

    # train_data, dev_data = handler.get_data_set_by_id(handler.LEAF_NODE_FT)
    # print(len(train_data))
    # print(len(dev_data))

    #
    #
    # data_set = generator.get_data_set("TRAIN", num_negative_samples=5)
    # export_data_set(data_set, "_tmp/test2.tsv")
    #
    # data_set = generator.get_data_set("DEV")
    # export_data_set(data_set, "_tmp/test3.tsv")
    #
    # data_set = generator.get_data_set("DEV")
    # export_data_set(data_set, "_tmp/test4.tsv")



