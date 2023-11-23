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
    PeftType,
    PeftConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
    IA3Config,
)

from data import (
    TASK_MAPPING_DATASET_ARGUMENTS,
    TASK_CONTAINS_TEST_SET,
    TASK_TO_SPLIT_MAPPING,
    TASK_TO_METRIC,
)


GLUE_TASKS = [
    "cola",
    "mnli",
    # "mnli-mm",
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
    # "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


if __name__ == "__main__":
    # write the code to argparse the most common parameters for training a bert model
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="debug")
    parser.add_argument("--task", type=str, default="cola")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--peft_model_id", type=str, default=None)
    parser.add_argument("--peft", type=str, default="")
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--test", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    args = parser.parse_args()
    print(args)

    os.environ["WANDB_PROJECT"] = args.project_name
    model_name = args.model_name_or_path.split("/")[-1]
    run_name = (
        f"{model_name}-{args.task}-epochs-{args.num_epochs}-lr-{args.learning_rate}"
        if args.peft == ""
        else f"{model_name}-{args.task}-{args.peft}-epochs-{args.num_epochs}-lr-{args.learning_rate}"
    )
    savedir = os.path.join(
        "/nas-ssd2/prateek/projects/peft_pruning/saved_runs",
        args.project_name,
        run_name,
    )
    if os.path.exists(savedir):
        if (
            ("config.json" in os.listdir(savedir))
            or ("adapter_config.json" in os.listdir(savedir))
            or ("tokenizer.json" in os.listdir(savedir))
        ):
            print(f"Run {run_name} already exists. Skipping...\n\n")
            exit(0)

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
            "train": raw_train_dataset,
            "validation": raw_eval_dataset,
            "test": raw_test_dataset,
        }
    )
    print(f"Number of training examples: {len(dataset['train'])}")
    print(f"Number of validation examples: {len(dataset['validation'])}")
    print(f"Number of test examples: {len(dataset['test'])}")

    metric = evaluate.load("glue", actual_task)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    sentence1_key, sentence2_key = task_to_keys[args.task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, max_length=None)
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            max_length=None,
        )

    encoded_dataset = dataset.map(
        preprocess_function,
        batched=True,
    )

    num_labels = 3 if args.task.startswith("mnli") else 1 if args.task == "stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels
    )
    if args.train:
        if args.peft == "lora":
            print("\nTraining with LORA\n")
            peft_config = LoraConfig(
                task_type="SEQ_CLS",
                inference_mode=False,
                target_modules=["query", "key", "value", "dense"],
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
            )
        elif args.peft == "ia3":
            print("\nTraining with IA3\n")
            peft_config = IA3Config(task_type="SEQ_CLS", inference_mode=False)
        elif args.peft == "prompt":
            print("\nTraining with Prompt Tuning\n")
            peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        elif args.peft == "prefix":
            print("\nTraining with Prefix Tuning\n")
            peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
        else:
            peft_config = None

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    elif not args.train and args.test:
        config = PeftConfig.from_pretrained(args.peft_model_id)
        model = PeftModel.from_pretrained(model, args.peft_model_id)

    # Visualization(model).structure_graph()

    metric_name = (
        "pearson"
        if args.task == "stsb"
        else "matthews_correlation"
        if args.task == "cola"
        else "accuracy"
    )

    training_args = TrainingArguments(
        savedir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.06,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=run_name,
        save_safetensors=False,
        seed=args.seed,
        data_seed=args.data_seed,
        fp16=args.fp16,
        run_name=run_name,
        remove_unused_columns=True,
        report_to=args.report_to,
        dataloader_num_workers=4,
        save_total_limit=1,
    )
    # print(f"\nTraining arguments:\n{training_args}")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if args.task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=(
            encoded_dataset["test"]
            if (args.test and not args.train)
            else encoded_dataset["validation"]
        ),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    trainer.callback_handler.callbacks[1]._wandb.config.update(
        vars(args), allow_val_change=True
    )

    test_results = trainer.evaluate(
        eval_dataset=encoded_dataset["test"], metric_key_prefix="test"
    )
    print(test_results)
    print("Finished evaluating")
