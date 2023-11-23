# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

sys.path.insert(0, "/nas-ssd2/prateek/projects/peft_pruning/src/qlora")
from utils import *
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)
from datasets import load_dataset, Dataset
import evaluate

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        generation_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args)
    )
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    os.environ["WANDB_PROJECT"] = args.project_name
    adaptor_name = args.checkpoint_dir.split("/")[-1].split("_")[-1]
    args.run_name = f"{adaptor_name}-k-{args.k}-rf-{args.replace_factor}"
    training_args.run_name = args.run_name
    training_args.task = adaptor_name
    args.output_dir = join(args.output_dir, adaptor_name, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(args.output_dir):
        if "metrics.json" in os.listdir(args.output_dir):
            print(f"Run {args.output_dir} already exists. Skipping...\n\n")
            exit(0)

    model, tokenizer = get_accelerate_model(args, args.checkpoint_dir)

    model.config.use_cache = False
    print("loaded model")
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    if args.mmlu_dataset == "mmlu-zs":
        mmlu_dataset = load_dataset(
            "json",
            data_files={
                "eval": "/nas-ssd2/prateek/projects/peft_pruning/src/qlora/data/mmlu/zero_shot_mmlu_val.json",
                "test": "/nas-ssd2/prateek/projects/peft_pruning/src/qlora/data/mmlu/zero_shot_mmlu_test.json",
            },
        )
        mmlu_dataset = mmlu_dataset.remove_columns("subject")
    # MMLU Five-shot (Eval/Test only)
    elif args.mmlu_dataset == "mmlu" or args.mmlu_dataset == "mmlu-fs":
        mmlu_dataset = load_dataset(
            "json",
            data_files={
                "eval": "/nas-ssd2/prateek/projects/peft_pruning/src/qlora/data/mmlu/five_shot_mmlu_val.json",
                "test": "/nas-ssd2/prateek/projects/peft_pruning/src/qlora/data/mmlu/five_shot_mmlu_test.json",
            },
        )
        # mmlu_dataset = mmlu_dataset.remove_columns('subject')
    mmlu_dataset = mmlu_dataset[args.mmlu_split]
    if args.max_mmlu_samples is not None:
        mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
    abcd_idx = [
        tokenizer("A", add_special_tokens=False).input_ids[0],
        tokenizer("B", add_special_tokens=False).input_ids[0],
        tokenizer("C", add_special_tokens=False).input_ids[0],
        tokenizer("D", add_special_tokens=False).input_ids[0],
    ]
    accuracy = evaluate.load("accuracy")

    class MMLUEvalCallback(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, model, **kwargs):
            data_loader = trainer.get_eval_dataloader(mmlu_dataset)
            source_max_len = trainer.data_collator.source_max_len
            trainer.data_collator.source_max_len = args.mmlu_source_max_len
            trainer.model.eval()
            preds, refs = [], []
            loss_mmlu = 0
            for batch in tqdm(data_loader, total=len(data_loader)):
                (loss, logits, labels) = trainer.prediction_step(
                    trainer.model,
                    batch,
                    prediction_loss_only=False,
                )
                # There are two tokens, the output, and eos token.
                for i, logit in enumerate(logits):
                    label_non_zero_id = (batch["labels"][i] != -100).nonzero()[0][0]
                    logit_abcd = logit[label_non_zero_id - 1][abcd_idx]
                    preds.append(torch.argmax(logit_abcd).item())
                labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
                refs += [abcd_idx.index(label) for label in labels.tolist()]
                loss_mmlu += loss.item()
            # Extract results by subject.
            results = {"mmlu_loss": loss_mmlu / len(data_loader)}
            subject = mmlu_dataset["subject"]
            subjects = {s: {"refs": [], "preds": []} for s in set(subject)}
            for s, p, r in zip(subject, preds, refs):
                subjects[s]["preds"].append(p)
                subjects[s]["refs"].append(r)
            subject_scores = []
            for subject in subjects:
                subject_score = accuracy.compute(
                    references=subjects[subject]["refs"],
                    predictions=subjects[subject]["preds"],
                )["accuracy"]
                results[f"mmlu_{args.mmlu_split}_accuracy_{subject}"] = subject_score
                subject_scores.append(subject_score)
            results[f"mmlu_{args.mmlu_split}_accuracy"] = np.mean(subject_scores)
            trainer.log(results)
            trainer.mmlu_results = results
            trainer.data_collator.source_max_len = source_max_len

    trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(metric_key_prefix=args.mmlu_split)
    mmlu_metrics = trainer.mmlu_results
    print(mmlu_metrics)
    trainer.log_metrics(args.mmlu_split, mmlu_metrics)
    # trainer.save_metrics("eval", mmlu_metrics)
    all_metrics.update(mmlu_metrics)

    if args.do_train or args.do_eval or args.do_predict:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
