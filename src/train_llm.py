import json
import os
import sys
import numpy as np

import backoff
import torch
import torch.distributed as dist
from datasets import Dataset
#from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
#from trl import SFTTrainer, SFTConfig

from validate import TrainingConfig

#SAVE_STEPS_LIST = np.array(7500 * np.array((0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1.0)), dtype=int)
SAVE_STEPS_LIST = np.array(5000 * np.array((0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)), dtype=int)

class IrregularSaveCallback(TrainerCallback):
    def __init__(self, save_steps_list):
        self.save_steps_list = set(save_steps_list)

    def on_step_end(self, args, state, control, **kwargs):
        # state.global_step is the current step
        if state.global_step in self.save_steps_list:
            control.should_save = True  # triggers save
        return control

def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]

def process_classification(df):
    def format_example(example):
        return {
            "description": example["description"],  # or however your text is stored
            "label": example["label"] - 1   # integer label (ensure they are in range [0, num_labels - 1])
        }
    
    df = df.map(format_example)
    return df


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""

    if rank := os.environ.get("LOCAL_RANK"):
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    else:
        rank = 0

    #print("Creating new LoRA adapter")
    #target_modules = training_cfg.target_modules
    #model = AutoModelForCausalLM.from_pretrained(
    #    training_cfg.model,
    #    device_map={"": f"cuda:{rank}"},
    #    quantization_config=BitsAndBytesConfig(
    #        load_in_8bit=True,
    #    ),
    #)

    ###
    # change to classificaiton model without lora
    model = AutoModelForSequenceClassification.from_pretrained(
        training_cfg.model,
        num_labels=training_cfg.num_labels,  # Add this to your config
        device_map={"": f"cuda:{rank}"},
        # Remove quantization_config for full training
        )
    ###

    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.model,
        token=os.environ.get("HF_TOKEN"),
        max_length=256,
    )

    # Use existing EOS token as pad token (no new token added)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Now set it in the model config
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Model embedding size: {model.base_model.embed_tokens.num_embeddings if hasattr(model.base_model, 'embed_tokens') else 'N/A'}")
    print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"Model pad token ID: {model.config.pad_token_id}")

    #tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # add padding token
    #new_vocab_size = len(tokenizer) # Get vocabulary size
    #model.resize_token_embeddings(new_vocab_size) # Update vocabulary size

    def tokenize_function(examples):
        # Assumes your data has 'text' and 'label' fields
        tokenized = tokenizer(
            examples["description"],  # Adjust field name to match your data
            padding="max_length",
            truncation=True,
            max_length=training_cfg.max_seq_length,
        )
        tokenized["labels"] = examples["label"]  # Add labels
        return tokenized

    # Remove for now (no LoRA)
    # Prepare for k-bit training
    #model = prepare_model_for_kbit_training(model)

    # 3. Define LoRA config
    #peft_config = LoraConfig(
    #    r=training_cfg.r,
    #    lora_alpha=training_cfg.lora_alpha,
    #    target_modules=target_modules,
    #    lora_dropout=training_cfg.lora_dropout,
    #    use_rslora=training_cfg.use_rslora,
    #    bias=training_cfg.lora_bias,
    #    task_type="CAUSAL_LM"
    #)
    dataset = Dataset.from_json(training_cfg.training_file)
    dataset = process_classification(dataset)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    if training_cfg.test_file:
        test_dataset = Dataset.from_json(training_cfg.test_file)
        test_dataset = process_classification(test_dataset)
        test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        #dataset = split["train"]
        test_dataset = split["test"]

    if training_cfg.order_by_attribution:
        xpl = torch.load(training_cfg.attribution_file)
        xpl_sum = xpl.sum(dim=0)
        sorted_idx = torch.argsort(xpl_sum, dim=-1, descending=training_cfg.descending_attribution, stable=False)
        dataset = dataset.select(sorted_idx)
        print(f"Argmax idx: {sorted_idx[0]} | Maximal attribution value: {xpl_sum[sorted_idx[0]]}")
    else:
        dataset = dataset.shuffle(seed=training_cfg.seed)

    #trainer = SFTTrainer(
    #    model=model,
    #    train_dataset=dataset,
    #    args=SFTConfig(
    #        completion_only_loss=True,
    #        ddp_find_unused_parameters=False,
    #        fp16=True,
    #         gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
    #         learning_rate=training_cfg.learning_rate,
    #         logging_steps=1,
    #         lr_scheduler_type=training_cfg.lr_scheduler_type,
    #         max_length=training_cfg.max_seq_length,
    #         max_steps=training_cfg.max_steps,
    #         num_train_epochs=training_cfg.epochs,
    #         label_names=["labels"],
    #         optim=training_cfg.optim,
    #         output_dir=training_cfg.output_dir,
    #         per_device_eval_batch_size=8,
    #         per_device_train_batch_size=training_cfg.per_device_train_batch_size,
    #         report_to=None,
    #         save_steps=training_cfg.save_steps,
    #         seed=training_cfg.seed,
    #         warmup_steps=training_cfg.warmup_steps,
    #         weight_decay=training_cfg.weight_decay,
    #     ),
    #     peft_config=peft_config,
    #     callbacks=[],
    #     eval_dataset=test_dataset,
    # )

    def accuracy_score(labels, predictions):
        correct = (predictions == labels).sum()
        total = len(labels)
        return correct / total

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # predictions: raw logits from model [batch_size, num_labels]
        # labels: true labels [batch_size]
        
        # Convert logits to class predictions
        predictions = predictions.argmax(axis=-1)
        
        # Calculate metrics
        return {
            "accuracy": accuracy_score(labels, predictions),
        }
    
    print(f"Label range: {min(dataset['labels'])} to {max(dataset['labels'])}")
    print(f"Num labels configured: {training_cfg.num_labels}")
    assert min(dataset['labels']) >= 0, "Labels contain negative values!"
    assert max(dataset['labels']) < training_cfg.num_labels, "Labels exceed num_labels!"

    # GPT2 for Sequence Classification uses CrossEntropyLoss for Single-Label-Classification
    trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=training_cfg.output_dir,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=training_cfg.epochs,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        warmup_steps=training_cfg.warmup_steps,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        eval_steps=training_cfg.save_steps,
        eval_strategy=training_cfg.eval_strategy,
        save_total_limit=None,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        fp16=True,
        seed=training_cfg.seed,
        ddp_find_unused_parameters=False,
    ),
    train_dataset=dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[IrregularSaveCallback(SAVE_STEPS_LIST)]
)
    
    trainer.train()

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")
    
    if rank == 0:
        finetuned_model_id = training_cfg.finetuned_model_id
        push_model(training_cfg, finetuned_model_id, model, tokenizer)
    
    if dist.is_initialized():
        dist.destroy_process_group()


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
# def push_model(training_cfg, finetuned_model_id, model, tokenizer):
#     if training_cfg.merge_before_push:
#         model.push_to_hub_merged(finetuned_model_id, tokenizer, save_method = "merged_16bit", token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
#     else:
#         model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
#         tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg.push_to_private)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    model.push_to_hub(
        finetuned_model_id, 
        token=os.environ['HF_TOKEN'], 
        private=training_cfg.push_to_private
    )
    tokenizer.push_to_hub(
        finetuned_model_id, 
        token=os.environ['HF_TOKEN'], 
        private=training_cfg.push_to_private
    )


def main(config: str):
    with open(config, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])