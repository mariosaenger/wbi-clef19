import argparse
import json
import random
import numpy as np

from collections import defaultdict
from math import ceil
from random import sample

from sklearn.linear_model import LogisticRegression
from typing import Dict, List

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

random.seed(42)


def collect_predictions(all_predictions, pad=False):
    collected_predictions = {}
    for predictions in all_predictions:
        for id_, preds in predictions.items():
            if id_ not in collected_predictions:
                collected_predictions[id_] = defaultdict(list)
            for concept, value in preds.items():
                collected_predictions[id_][concept].append(value)
    return collected_predictions


def predict_ensemble_mean(all_predictions: List[Dict[str, Dict[str, float]]]):
    collected_predictions = collect_predictions(all_predictions)
    final_predictions = dict()
    for id_, preds in collected_predictions.items():
        final_predictions[id_] = list()
        for concept, values in preds.items():
            if "NoLabel" in concept:
                continue

            if np.mean(values) > 0.5:
                final_predictions[id_].append(concept)

    return final_predictions


def predict_ensemble_clf(test_predictions: List[Dict[str, Dict[str, float]]],
                         dev_predictions: List[Dict[str, Dict[str, float]]],
                         true_labels: Dict[str, List[str]]):
    assert len(test_predictions) == len(dev_predictions)
    n_models = len(test_predictions)

    dev_predictions = collect_predictions(dev_predictions)

    instances_dict = {}

    for id_, preds in dev_predictions.items():
        gold_labels = true_labels.get(id_, [])

        pos_instances = []
        neg_instances = []

        for concept, values in preds.items():
            assert len(values) == n_models
            if concept in gold_labels:
                pos_instances.append(values)
            else:
                neg_instances.append(values)

        instances_dict[id_] = (pos_instances, neg_instances)

    print(f"Found {len(instances_dict)} instances in total")

    num_val_instances = ceil(len(instances_dict) * 0.25)
    val_instances = sample(instances_dict.items(), num_val_instances)
    #val_instances = instances_dict.items()

    X_val = []
    y_val = []

    print(f"Building {len(val_instances)} validation instances")
    for (id, (pos_instances, neg_instances)) in val_instances:
        for pos_instance in pos_instances:
            X_val.append(pos_instance)
            y_val.append(1)

        for neg_instance in neg_instances:
            X_val.append(neg_instance)
            y_val.append(0)

        instances_dict.pop(id)

    print(f"Using {len(instances_dict)} instances for training")

    max_f1 = 0
    best_clf = None
    best_size = 0

    neg_sample_sizes = [30, 35, 40]
    for neg_sample_size in tqdm(neg_sample_sizes, total=len(neg_sample_sizes)):
        X_train = []
        y_train = []

        for (_, (pos_instances, neg_instances)) in instances_dict.items():

            for instance in pos_instances:
                X_train.append(instance)
                y_train.append(1)

            for instance in sample(neg_instances, neg_sample_size):
                X_train.append(instance)
                y_train.append(0)

        parameter_grid = {"C": [2 ** i for i in range(-5, 17, 2)]}
        grid_search = GridSearchCV(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=100000, n_jobs=4),
                                   parameter_grid, scoring="f1", n_jobs=16, cv=5, verbose=100)
        clf = grid_search.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        if f1 > max_f1:
            print(f"Found new best f1 {f1} with C={clf.best_params_['C']} and neg_sample_size={neg_sample_size}")
            max_f1 = f1
            best_clf = clf
            best_size = neg_sample_size

    print(f"Found best f1 of {max_f1} with C={best_clf.best_params_['C']} and neg_sample_size={best_size}")

    print("Start prediction 1")
    test_predictions = collect_predictions(test_predictions)
    final_predictions = dict()
    for id_, preds in tqdm(test_predictions.items(), total=len(test_predictions)):
        final_predictions[id_] = list()
        for concept, values in preds.items():
            assert len(values) == n_models

            if "NoLabel" in concept:
                continue

            if best_clf.predict([values]).squeeze():
                final_predictions[id_].append(concept)

    X_train = []
    y_train = []

    print(f"Building full training set")
    for (_, (pos_instances, neg_instances)) in val_instances:
        for pos_instance in pos_instances:
            X_train.append(pos_instance)
            y_train.append(1)

        for neg_instance in sample(neg_instances, best_size):
            X_train.append(neg_instance)
            y_train.append(0)

    for (_, (pos_instances, neg_instances)) in instances_dict.items():
        for pos_instance in pos_instances:
            X_train.append(pos_instance)
            y_train.append(1)

        for neg_instance in sample(neg_instances, best_size):
            X_train.append(neg_instance)
            y_train.append(0)

    best_clf = LogisticRegression(C=best_clf.best_params_["C"], penalty="l2", solver="lbfgs", max_iter=100000, n_jobs=4)
    best_clf.fit(X_train, y_train)

    print("Start prediction 2")
    #test_predictions = collect_predictions(test_predictions)
    final_predictions2 = dict()
    for id_, preds in tqdm(test_predictions.items(), total=len(test_predictions)):
        final_predictions2[id_] = list()
        for concept, values in preds.items():
            assert len(values) == n_models

            if "NoLabel" in concept:
                continue

            if best_clf.predict([values]).squeeze():
                final_predictions2[id_].append(concept)

    return final_predictions, final_predictions2


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Compute ensemble predictions.
    dev_predictions and test_predictions have to be in the same order.
    all prediction files have to contain predictions for all concepts and for all document ids.
    """)
    parser.add_argument('--dev_predictions', nargs='+')
    parser.add_argument('--test_predictions', nargs='+')
    parser.add_argument('--anns', required=True)
    parser.add_argument('--method', required=True, choices=['clf', 'mean'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    test_predictions = [json.load(open(pred)) for pred in args.test_predictions]

    if args.method == 'mean':
        ensemble_preds = predict_ensemble_mean(test_predictions)
    elif args.method == 'clf':
        dev_predictions = [json.load(open(pred)) for pred in args.dev_predictions]
        anns = {}
        with open(args.anns) as f:
            for line in f:
                id_, labels = line.strip().split('\t')
                anns[id_] = labels.split('|')
        ensemble_preds, ensemble_preds2 = predict_ensemble_clf(test_predictions=test_predictions,
                                              dev_predictions=dev_predictions,
                                              true_labels=anns)
        with open(args.output+"2", 'w') as f:
            for key, value in ensemble_preds2.items():
                if len(value) > 0:
                    labels = "|".join(value)
                    f.write(f"{key}\t{labels}\n")
                else:
                    f.write(f"{key}\n")

    with open(args.output, 'w') as f:
        for key, value in ensemble_preds.items():
            if len(value) > 0:
                labels = "|".join(value)
                f.write(f"{key}\t{labels}\n")
            else:
                f.write(f"{key}\n")

