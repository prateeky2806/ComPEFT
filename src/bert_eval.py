import os, argparse
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from bigmodelvis import Visualization

import datasets, evaluate
import pandas as pd, numpy as np

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftConfig,
    PeftModel,
)
from utils import *
from data import (
    TASK_MAPPING_DATASET_ARGUMENTS,
    TASK_CONTAINS_TEST_SET,
    TASK_TO_SPLIT_MAPPING,
    TASK_TO_METRIC,
)

GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def compress_and_update_parms(model, k, replace_factor, peft, ptm_base_sd=None):
    if k == 100:
        return

    if peft == "ia3":
        weights = get_peft_model_state_dict(model)
        remove_keys = []
        flat_ptm = 1
    elif peft == "lora":
        weights = get_peft_model_state_dict(model)
        remove_keys = []
        flat_ptm = 0
    elif peft == "":
        weights = model.state_dict()
        remove_keys = [
            "classifier.bias",
            "classifier.weight",
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        ]
        flat_ptm = state_dict_to_vector(ptm_base_sd, remove_keys=remove_keys)

    flat_weights = state_dict_to_vector(weights, remove_keys=remove_keys)
    tv = flat_weights - flat_ptm
    mean, std = tv.mean(), tv.std()
    topk_flat_tv, topk_mask = topk_values_mask(tv, K=k, return_mask=True)
    assert tv[~topk_mask].abs().max() <= topk_flat_tv[topk_mask].abs().min()

    if replace_factor == -2:
        print("Using STC compression!")
        alpha = topk_flat_tv[topk_flat_tv != 0].mean()
        updated_flat_tv = alpha * topk_flat_tv.sign()
    elif replace_factor > 0:
        print("Using Com2PEFT Compression!")
        alpha = replace_factor * std
        updated_flat_tv = alpha * topk_flat_tv.sign()
    elif replace_factor == -1:
        print("Using Just Pruning!")
        updated_flat_tv = topk_flat_tv

    updated_flat_weights = updated_flat_tv + flat_ptm
    updated_weights = vector_to_state_dict(
        updated_flat_weights, weights, remove_keys=remove_keys
    )
    if peft != "":
        missing_keys, unexpected_keys = set_peft_model_state_dict(
            model, updated_weights
        )
        assert set(missing_keys) - set(weights.keys()) == set(missing_keys)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(
            updated_weights, strict=False
        )
        assert set(missing_keys).issubset(set(remove_keys))
        assert len(unexpected_keys) == 0


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if args.task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, max_length=None)
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        truncation=True,
        max_length=None,
    )


if __name__ == "__main__":
    # write the code to argparse the most common parameters for training a bert model
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="debug")
    parser.add_argument("--task", type=str, default="cola")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--checkpoint_load_path", type=str, default=None)
    parser.add_argument("--peft", type=str, default="")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--replace_factor", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--data_seed", type=int, default=6)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    args = parser.parse_args()
    print(args)

    os.environ["WANDB_PROJECT"] = args.project_name
    model_name = args.model_name_or_path.split("/")[-1]
    run_name = (
        f"{model_name}-{args.task}-k-{args.k}-replacefactor-{args.replace_factor}"
        if args.peft == ""
        else f"{model_name}-{args.task}-{args.peft}-k-{args.k}-replacefactor-{args.replace_factor}"
    )
    savedir = os.path.join(
        "/nas-ssd2/prateek/projects/peft_pruning/saved_runs",
        args.project_name,
        run_name,
    )

    training_args = TrainingArguments(
        savedir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        push_to_hub=args.push_to_hub,
        seed=args.seed,
        data_seed=args.data_seed,
        fp16=args.fp16,
        run_name=run_name,
        remove_unused_columns=True,
        report_to=args.report_to,
        dataloader_num_workers=4,
    )
    print(f"\nTraining arguments:\n{training_args}")

    # Load and split datasets
    actual_task = args.task
    raw_dataset = datasets.load_dataset("glue", actual_task)
    raw_train_dataset = raw_dataset[TASK_TO_SPLIT_MAPPING[actual_task]["train"]]
    raw_eval_dataset = raw_dataset[TASK_TO_SPLIT_MAPPING[actual_task]["validation"]]
    if TASK_CONTAINS_TEST_SET[actual_task][0]:
        raw_test_dataset = raw_dataset["test"]
    else:
        if TASK_CONTAINS_TEST_SET[actual_task][1] == "validation":
            # split validation dataset into validation and test
            split_datasets = raw_eval_dataset.train_test_split(
                test_size=0.5, seed=args.data_seed
            )
            raw_eval_dataset = split_datasets["train"]
            raw_test_dataset = split_datasets["test"]
        elif TASK_CONTAINS_TEST_SET[actual_task][1] == "train":
            # split the train dataset into train and test
            split_datasets = raw_train_dataset.train_test_split(
                test_size=TASK_CONTAINS_TEST_SET[actual_task][2], seed=args.data_seed
            )
            raw_train_dataset = split_datasets["train"]
            raw_test_dataset = split_datasets["test"]

    dataset = datasets.DatasetDict(
        {
            "validation": raw_eval_dataset,
            "test": raw_test_dataset,
        }
    )
    print(f"Number of test examples: {len(dataset['validation'])}")
    print(f"Number of test examples: {len(dataset['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    sentence1_key, sentence2_key = task_to_keys[args.task]
    tokenied_dataset = dataset.map(
        preprocess_function,
        batched=True,
    )
    metric = evaluate.load("glue", actual_task)

    num_labels = 3 if args.task.startswith("mnli") else 1 if args.task == "stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels
    )

    if args.peft != "":
        ptm_base_sd = None
        config = PeftConfig.from_pretrained(args.checkpoint_load_path)
        model = PeftModel.from_pretrained(model, args.checkpoint_load_path)
    else:
        ptm_base_sd = model.state_dict()
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint_load_path, num_labels=num_labels
        )

    compress_and_update_parms(
        model, args.k, args.replace_factor, args.peft, ptm_base_sd=ptm_base_sd
    )
    Visualization(model).structure_graph()

    trainer = Trainer(
        model,
        training_args,
        eval_dataset=tokenied_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Evaluate on evaluate set!".upper())
    eval_results = trainer.evaluate(
        eval_dataset=tokenied_dataset["validation"], metric_key_prefix="eval"
    )
    print(eval_results)
    trainer.callback_handler.callbacks[1]._wandb.config.update(
        args, allow_val_change=True
    )
    print("Evaluate on test set!".upper())
    test_results = trainer.evaluate(
        eval_dataset=tokenied_dataset["test"], metric_key_prefix="test"
    )
    print(test_results)

    # test_results = trainer.evaluate()
    # trainer.callback_handler.callbacks[1]._wandb.config.update(
    #     vars(args), allow_val_change=True
    # )
    # print(test_results)
    # print("Finished evaluating")
