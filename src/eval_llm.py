import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    DataCollatorWithPadding,
)
import torch

def accuracy_score(labels, predictions):
    correct = (predictions == labels).sum()
    total = len(labels)
    return correct / total

# --- Configuration (Must match your training setup) ---
# IMPORTANT: Adjust these paths to match your actual environment
MODEL_BASE_DIR = "/home/weckbecker/coding/DualXDA/src/models/gpt2-ag_news/train_full"
TEST_FILE_PATH = "/home/weckbecker/coding/DualXDA/src/dataset/ag_news/test.jsonl"
OUTPUT_PLOTS_DIR = os.path.join(MODEL_BASE_DIR, "evaluation_plots")
OUTPUT_RESULTS_FILE = os.path.join(MODEL_BASE_DIR, "checkpoint_evaluation_results.csv")

# NOTE: Since the tokenizer uses 'gpt2' with specific settings (pad_token='<|endoftext|>'), 
# we use the base name here. The actual model is loaded from the checkpoint path.
MODEL_NAME_OR_PATH = "gpt2" 
NUM_LABELS = 4 # AG News has 4 classes (0 to 3, since you do label - 1)
MAX_SEQ_LENGTH = 256 # Adjust this to match your training max_seq_length

# Create output directory
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# --------------------------
# 1. & 2. Find Checkpoints and Extract Step Numbers
# --------------------------

def get_checkpoints(base_dir):
    """Finds all 'checkpoint-xxx' directories and returns their paths and step numbers."""
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []
    
    for entry in os.listdir(base_dir):
        match = checkpoint_pattern.match(entry)
        if match:
            step = int(match.group(1))
            path = os.path.join(base_dir, entry)
            checkpoints.append((step, path))

    # Sort checkpoints by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

# --------------------------
# 3. & 4. Setup Evaluation and Compute Metrics/Data Preparation
# --------------------------

# a) Define the metric function (matching the one in your train script)
def compute_metrics(p: EvalPrediction):
    """Computes accuracy for sequence classification."""
    predictions, labels = p
    # Convert logits to class predictions
    predictions = predictions.argmax(axis=-1)
    # Use sklearn's accuracy_score for consistency with standard metrics
    return {"accuracy": accuracy_score(labels, predictions)}

# b) Load Tokenizer and Dataset
try:
    # Load the tokenizer with the same settings used during training
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, 
        use_fast=True,
        pad_token='<|endoftext|>'
    )
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)


# Load and preprocess the test dataset (matching your train script's logic)
def load_and_preprocess_test_data(file_path, tokenizer, max_len):
    """Loads, formats, and tokenizes the test data."""
    # 1. Load from jsonl
    raw_test_data = [json.loads(line) for line in open(file_path, "r").readlines() if line.strip()]
    raw_test_dataset = Dataset.from_list(raw_test_data)
    
    # 2. Format: label - 1
    def format_example(example):
        # Assumes your data has 'description' and 'label'
        return {"description": example["description"], "label": example["label"] - 1}

    formatted_dataset = raw_test_dataset.map(format_example)

    # 3. Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["description"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    tokenized_dataset = formatted_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=formatted_dataset.column_names
    )
    
    # Set format for Trainer
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

tokenized_test_dataset = load_and_preprocess_test_data(
    TEST_FILE_PATH, 
    tokenizer, 
    MAX_SEQ_LENGTH
)

# c) Setup Trainer arguments
training_args = TrainingArguments(
    output_dir="./evaluation_tmp",  # Temporary directory for Trainer output
    per_device_eval_batch_size=32,
    do_eval=True,
    report_to="none",
)

# --------------------------
# 5. Loop Through Checkpoints and Evaluate
# --------------------------

def evaluate_checkpoints():
    checkpoint_results = []
    checkpoints = get_checkpoints(MODEL_BASE_DIR)

    if not checkpoints:
        print(f"No checkpoint directories found in {MODEL_BASE_DIR}. Please check the path.")
        return pd.DataFrame()

    print(f"Found {len(checkpoints)} checkpoints. Starting evaluation...")
    
    for step, checkpoint_path in checkpoints:
        print(f"\nEvaluating checkpoint-{step}...")
        
        try:
            # Load the model from the checkpoint
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path, 
                num_labels=NUM_LABELS
            )
            
            # Re-initialize the Trainer for each checkpoint with the new model
            trainer = Trainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics,
            )

            # Run evaluation
            metrics = trainer.evaluate(tokenized_test_dataset)

            # Extract desired metrics
            loss = metrics.get('eval_loss')
            accuracy = metrics.get('eval_accuracy')

            # Store results
            checkpoint_results.append({
                "step": step,
                "checkpoint_path": checkpoint_path,
                "test_loss": loss,
                "test_accuracy": accuracy,
            })
            
            print(f"  -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Clear GPU memory if possible (optional but recommended)
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"An error occurred while processing checkpoint-{step}: {e}")

    # Save results to a CSV file
    results_df = pd.DataFrame(checkpoint_results).sort_values(by="step")
    results_df.to_csv(OUTPUT_RESULTS_FILE, index=False)
    print(f"\nEvaluation results saved to {OUTPUT_RESULTS_FILE}")
    
    return results_df

results_df = evaluate_checkpoints()

# --------------------------
# 6. Plotting the Results
# --------------------------

if not results_df.empty:
    
    # Plot 1: Loss vs. Step
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["step"], results_df["test_loss"], marker='o', linestyle='-', color='tab:blue')
    plt.title(f'Test Loss vs. Checkpoint Step (Model: {os.path.basename(MODEL_BASE_DIR)})')
    plt.xlabel('Checkpoint Step Number')
    plt.ylabel('Test Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    loss_plot_path = os.path.join(OUTPUT_PLOTS_DIR, "test_loss_vs_step.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Test Loss plot saved to {loss_plot_path}")

    # Plot 2: Accuracy vs. Step
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["step"], results_df["test_accuracy"], marker='o', linestyle='-', color='tab:green')
    plt.title(f'Test Accuracy vs. Checkpoint Step (Model: {os.path.basename(MODEL_BASE_DIR)})')
    plt.xlabel('Checkpoint Step Number')
    plt.ylabel('Test Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    accuracy_plot_path = os.path.join(OUTPUT_PLOTS_DIR, "test_accuracy_vs_step.png")
    plt.savefig(accuracy_plot_path)
    plt.close()
    print(f"Test Accuracy plot saved to {accuracy_plot_path}")
else:
    print("No results to plot.")