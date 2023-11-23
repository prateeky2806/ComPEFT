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

logger = logging.get_logger(__name__)
# logger.setLevel(20)

MODEL_TO_LORA_MODULES = {
    "t5-base": ["q", "k", "v", "o", "wi", "wo"],
    "t5-large": ["q", "k", "v", "o", "wi", "wo"],
    "google/t5-v1_1-base": ["q", "k", "v", "o", "wi", "wo"],
    "google/t5-v1_1-large": ["q", "k", "v", "o", "wi", "wo"],
    "bigscience/T0_3B": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
}


def finetune_on_downstream_task(
    project_name: Optional[str] = "debug",
    model_name_or_path: Optional[str] = "t5-base",
    peft: Optional[str] = "",
    train: Optional[bool] = True,
    test: Optional[bool] = True,
    task: Optional[str] = "cola",
    per_device_train_batch_size: Optional[int] = 16,
    per_device_eval_batch_size: Optional[int] = 16,
    num_epochs: Optional[int] = 10,
    logging_steps: Optional[int] = 10,
    warmup_ratio: Optional[float] = 0.06,
    weight_decay: Optional[float] = 0.01,
    learning_rate: Optional[float] = 5e-4,
    gradient_accumulation_steps: Optional[int] = 1,
    preprocessing_num_workers: Optional[int] = 8,
    seed: Optional[int] = 42,
    data_seed: Optional[int] = 42,
    fp16: Optional[bool] = False,
    push_to_hub: Optional[bool] = False,
    hub_private_repo=True,
    report_to: Optional[str] = "wandb",
):
    config = copy.deepcopy(locals())

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
    if os.path.exists(savedir) and train:
        if ("config.json" in os.listdir(savedir)) or (
            "adapter_config.json" in os.listdir(savedir)
        ):
            print(f"Run {run_name} already exists. Skipping...\n\n")
            exit(0)

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
    train_dataset = raw_train_dataset
    eval_dataset = raw_eval_dataset
    test_dataset = raw_test_dataset

    # Apply templates and preprocess datasets
    if task == "hotpotqa":
        train_dataset = train_dataset.map(
            keep_only_supporting_facts_in_context_for_hotpotqa,
            batched=False,
            num_proc=preprocessing_num_workers,
        )
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
    train_dataset = train_dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task),
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task, keep_specific_keys=EXTRA_KEYS_FOR_EVAL),
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
    )

    # tokenize datasets
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=False),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=True),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
    )
    print(f"Number of training examples: {len(tokenized_train_dataset)}")
    print(f"Number of validation examples: {len(tokenized_eval_dataset)}")

    test_dataset = test_dataset.map(
        Seq2SeqDataPreProcessor(benchmark=task, keep_specific_keys=EXTRA_KEYS_FOR_EVAL),
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=test_dataset.column_names,
    )
    tokenized_test_dataset = test_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=True),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=False,
    )
    print(f"Number of test examples: {len(tokenized_test_dataset)}")

    # Load model and Peft parameters.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    if peft == "lora":
        print("\nTraining with LORA\n")
        peft_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
            target_modules=MODEL_TO_LORA_MODULES[model_name_or_path],
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
    elif peft == "ia3":
        print("\nTraining with IA3\n")
        peft_config = IA3Config(task_type="SEQ_2_SEQ_LM", inference_mode=False)
    elif peft == "prompt":
        print("\nTraining with Prompt Tuning\n")
        peft_config = PromptTuningConfig(
            task_type="SEQ_2_SEQ_LM", num_virtual_tokens=10
        )
    elif peft == "prefix":
        print("\nTraining with Prefix Tuning\n")
        peft_config = PrefixTuningConfig(
            task_type="SEQ_2_SEQ_LM", num_virtual_tokens=20
        )
    else:
        peft_config = None
        print(f"Number of parameters in T5: {model.num_parameters()}")

    if peft_config is not None:
        model = get_peft_model(model, peft_config)
        print(
            f"Num. Trainable: {model.print_trainable_parameters()}\tNum. Total: {model.num_parameters()}"
        )

    # Visualization(model).structure_graph()

    metric_name = TASK_TO_METRIC[task]
    # define training arguments
    training_args = Seq2SeqTrainingArguments(
        savedir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type="linear",
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        push_to_hub=push_to_hub,
        hub_private_repo=hub_private_repo,
        hub_model_id=run_name,
        save_safetensors=True,
        logging_steps=logging_steps,
        seed=seed,
        data_seed=data_seed,
        fp16=fp16,
        run_name=run_name,
        remove_unused_columns=False,
        report_to=report_to,
        dataloader_num_workers=preprocessing_num_workers,
        save_total_limit=1,
    )
    # print(f"\nTraining arguments:\n{training_args}")

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
        train_dataset=tokenized_train_dataset,
        eval_dataset=(
            tokenized_test_dataset if (test and not train) else tokenized_eval_dataset
        ),
        data_collator=data_collator,
        compute_metrics=evaluate_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("Start training!".upper())
    trainer.train()
    trainer.save_model()
    trainer.callback_handler.callbacks[1]._wandb.config.update(
        config, allow_val_change=True
    )

    print("Evaluate on test set!".upper())
    test_results = trainer.evaluate(
        eval_dataset=tokenized_test_dataset, metric_key_prefix="test"
    )
    print(test_results)


if __name__ == "__main__":
    Fire(finetune_on_downstream_task)
