import math
import os
import os.path
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import copy
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_scheduler,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    logging,
    set_seed,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PeftConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
    IA3Config,
)
from bigmodelvis import Visualization

from trainer import T5Trainer
from data import (
    Seq2SeqDataPreProcessor,
    tokenize_seq2seq,
    TASK_MAPPING_DATASET_ARGUMENTS,
    TASK_CONTAINS_TEST_SET,
    TASK_TO_SPLIT_MAPPING,
    TASK_TO_METRIC,
    DataCollatorForSeq2Seq,
    get_evaluate_fn,
    EXTRA_KEYS_FOR_EVAL,
    keep_only_supporting_facts_in_context_for_hotpotqa,
)
from utils import *

logger = logging.get_logger(__name__)


def compress_and_update_t5_parms(model, k, replace_factor, peft, ptm_base_sd=None):
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
            "encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight",
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
        for rk in remove_keys:
            if rk not in updated_weights:
                updated_weights[rk] = updated_weights["shared.weight"]
        missing_keys, unexpected_keys = model.load_state_dict(updated_weights)
    assert len(unexpected_keys) == 0


def compress_and_evaluate_downstream(
    project_name: Optional[str] = "debug",
    model_name_or_path: Optional[str] = "t5-base",
    peft: Optional[str] = "",
    checkpoint_load_path: Optional[str] = None,
    k: Optional[int] = 20,
    replace_factor: Optional[float] = -1,
    task: Optional[str] = "cola",
    per_device_train_batch_size: Optional[int] = 16,
    per_device_eval_batch_size: Optional[int] = 64,
    num_epochs: Optional[int] = 10,
    learning_rate: Optional[float] = 5e-4,
    preprocessing_num_workers: Optional[int] = 8,
    seed: Optional[int] = 42,
    data_seed: Optional[int] = 42,
    fp16: Optional[bool] = False,
    report_to: Optional[str] = "wandb",
):
    args = copy.deepcopy(locals())

    if task not in TASK_MAPPING_DATASET_ARGUMENTS:
        raise ValueError(
            f"task must be one of {list(TASK_MAPPING_DATASET_ARGUMENTS.keys())}"
        )

    set_seed(seed)
    os.environ["WANDB_PROJECT"] = project_name
    model_name = model_name_or_path.split("/")[-1]
    run_name = (
        f"{model_name}-{task}-epochs-{num_epochs}-lr-{learning_rate}"
        if peft == ""
        else f"{model_name}-{task}-{peft}-epochs-{num_epochs}-lr-{learning_rate}"
    )
    savedir = os.path.join(
        "/nas-ssd2/prateek/projects/peft_pruning/saved_runs",
        project_name,
        run_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        keys_to_ignore=EXTRA_KEYS_FOR_EVAL,
    )

    # Load and split datasets
    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])
    raw_train_dataset = raw_dataset[TASK_TO_SPLIT_MAPPING[task]["train"]]
    raw_eval_dataset = raw_dataset[TASK_TO_SPLIT_MAPPING[task]["validation"]]
    if TASK_CONTAINS_TEST_SET[task][0]:
        raw_test_dataset = raw_dataset["test"]
    else:
        if TASK_CONTAINS_TEST_SET[task][1] == "validation":
            # split validation dataset into validation and test
            split_datasets = raw_eval_dataset.train_test_split(
                test_size=0.5, seed=data_seed
            )
            raw_eval_dataset = split_datasets["train"]
            raw_test_dataset = split_datasets["test"]
        elif TASK_CONTAINS_TEST_SET[task][1] == "train":
            # split the train dataset into train and test
            split_datasets = raw_train_dataset.train_test_split(
                test_size=TASK_CONTAINS_TEST_SET[task][2], seed=data_seed
            )
            raw_train_dataset = split_datasets["train"]
            raw_test_dataset = split_datasets["test"]

    del raw_train_dataset
    eval_dataset = raw_eval_dataset
    test_dataset = raw_test_dataset

    # Apply templates and preprocess datasets
    if task == "hotpotqa":
        eval_dataset = eval_dataset.map(
            keep_only_supporting_facts_in_context_for_hotpotqa,
            batched=False,
            num_proc=preprocessing_num_workers,
        )
        test_dataset = test_dataset.map(
            keep_only_supporting_facts_in_context_for_hotpotqa,
            batched=False,
            num_proc=preprocessing_num_workers,
        )
    eval_dataset = eval_dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task, keep_specific_keys=EXTRA_KEYS_FOR_EVAL),
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task, keep_specific_keys=EXTRA_KEYS_FOR_EVAL),
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=test_dataset.column_names,
    )

    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=True),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
    )
    tokenized_test_dataset = test_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=True),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=False,
    )
    print(f"Number of validation examples: {len(tokenized_eval_dataset)}")
    print(f"Number of test examples: {len(tokenized_test_dataset)}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    if peft != "":
        ptm_base_sd = None
        config = PeftConfig.from_pretrained(checkpoint_load_path)
        model = PeftModel.from_pretrained(model, checkpoint_load_path)
    else:
        ptm_base_sd = model.state_dict()
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_load_path)

    compress_and_update_t5_parms(
        model, k, replace_factor, peft, ptm_base_sd=ptm_base_sd
    )
    Visualization(model).structure_graph()

    # define training arguments
    training_args = Seq2SeqTrainingArguments(
        savedir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        seed=seed,
        data_seed=data_seed,
        fp16=fp16,
        run_name=run_name,
        remove_unused_columns=False,
        report_to=report_to,
        dataloader_num_workers=preprocessing_num_workers,
    )
    print(f"\nTraining arguments:\n{training_args}")

    evaluate_fn = get_evaluate_fn(
        task=task,
        tokenizer=tokenizer,
        raw_eval_dataset=None,
    )

    def preprocess_logits_for_metrics(logits, labels):
        if type(logits) == type(tuple()):
            logits = logits[0]
        return logits.argmax(dim=-1)

    trainer = T5Trainer(
        model,
        training_args,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        compute_metrics=evaluate_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("Evaluate on evaluate set!".upper())
    eval_results = trainer.evaluate(
        eval_dataset=tokenized_eval_dataset, metric_key_prefix="eval"
    )
    print(eval_results)
    trainer.callback_handler.callbacks[1]._wandb.config.update(
        args, allow_val_change=True
    )
    print("Evaluate on test set!".upper())
    test_results = trainer.evaluate(
        eval_dataset=tokenized_test_dataset, metric_key_prefix="test"
    )
    print(test_results)


if __name__ == "__main__":
    Fire(compress_and_evaluate_downstream)
