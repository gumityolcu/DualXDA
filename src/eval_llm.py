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

# --- Configuration ---
BASE_DIR = "/home/weckbecker/coding/DualXDA/src/models/llama2-ag_news"
TEST_FILE_PATH = "/home/weckbecker/coding/DualXDA/src/dataset/ag_news/test.jsonl"
OUTPUT_RESULTS_FILE = os.path.join(BASE_DIR, "all_checkpoints_evaluation_results.csv")

MODEL_NAME_OR_PATH = "llama" 
NUM_LABELS = 4
MAX_SEQ_LENGTH = 128

# Only evaluate these specific checkpoints
TARGET_CHECKPOINTS = [7, 15, 37, 75, 150, 375, 750, 1500]

# --------------------------
# Helper Functions
# --------------------------

def load_existing_results(output_file):
    """Load existing results from CSV if it exists."""
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            # Remove duplicates based on model_folder and checkpoint
            df = df.drop_duplicates(subset=["model_folder", "checkpoint"], keep="last")
            print(f"Loaded existing results from {output_file}")
            print(f"Found {len(df)} existing evaluations")
            return df
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return pd.DataFrame()
    else:
        print(f"No existing results file found. Will create new file.")
        return pd.DataFrame()

def is_already_evaluated(existing_df, folder_name, checkpoint_step):
    """Check if a specific checkpoint has already been evaluated."""
    if existing_df.empty:
        return False
    
    mask = (existing_df["model_folder"] == folder_name) & (existing_df["checkpoint"] == checkpoint_step)
    return mask.any()

def get_checkpoints(base_dir, target_steps):
    """Finds checkpoint directories matching target steps and returns their paths and step numbers."""
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []
    
    if not os.path.exists(base_dir):
        return checkpoints
    
    for entry in os.listdir(base_dir):
        match = checkpoint_pattern.match(entry)
        if match:
            step = int(match.group(1))
            if step in target_steps:  # Only include target checkpoints
                path = os.path.join(base_dir, entry)
                checkpoints.append((step, path))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def get_model_folders(base_dir, target_steps):
    """Returns all subdirectories in base_dir that contain target checkpoints, sorted alphabetically."""
    model_folders = []
    
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist.")
        return model_folders
    
    for entry in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, entry)
        if os.path.isdir(folder_path):
            # Check if this folder contains any target checkpoints
            checkpoints = get_checkpoints(folder_path, target_steps)
            if checkpoints:
                model_folders.append((entry, folder_path))
    
    # Sort alphabetically by folder name
    model_folders.sort(key=lambda x: x[0])
    return model_folders

def compute_metrics(p: EvalPrediction):
    """Computes accuracy for sequence classification."""
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def load_and_preprocess_test_data(file_path, tokenizer, max_len):
    """Loads, formats, and tokenizes the test data."""
    raw_test_data = [json.loads(line) for line in open(file_path, "r").readlines() if line.strip()]
    raw_test_dataset = Dataset.from_list(raw_test_data)
    
    def format_example(example):
        return {"description": example["description"], "label": example["label"] - 1}

    formatted_dataset = raw_test_dataset.map(format_example)

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
    
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

# --------------------------
# Main Evaluation
# --------------------------

def evaluate_all_model_folders():
    """Evaluates all checkpoints across all model folders."""
    # Load existing results
    existing_results_df = load_existing_results(OUTPUT_RESULTS_FILE)
    new_results = []
    
    # Load tokenizer once
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME_OR_PATH, 
            use_fast=True,
            pad_token='<|endoftext|>'
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return pd.DataFrame()
    
    # Load test dataset once
    print("Loading test dataset...")
    tokenized_test_dataset = load_and_preprocess_test_data(
        TEST_FILE_PATH, 
        tokenizer, 
        MAX_SEQ_LENGTH
    )
    
    # Setup Trainer arguments
    training_args = TrainingArguments(
        output_dir="./evaluation_tmp",
        per_device_eval_batch_size=96,
        do_eval=True,
        report_to="none",
    )
    
    # Get all model folders
    model_folders = get_model_folders(BASE_DIR, TARGET_CHECKPOINTS)
    
    if not model_folders:
        print(f"No model folders with target checkpoints found in {BASE_DIR}")
        return existing_results_df
    
    print(f"\nFound {len(model_folders)} model folders with target checkpoints.")
    print(f"Target checkpoints: {TARGET_CHECKPOINTS}\n")
    
    # Iterate through each model folder
    for folder_name, folder_path in model_folders:
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*60}")
        
        checkpoints = get_checkpoints(folder_path, TARGET_CHECKPOINTS)
        print(f"Found {len(checkpoints)} target checkpoints in {folder_name}")
        
        # Evaluate each checkpoint in this folder
        for step, checkpoint_path in checkpoints:
            # Check if already evaluated
            if is_already_evaluated(existing_results_df, folder_name, step):
                print(f"\n  Checkpoint-{step} already evaluated. Skipping...")
                continue
            
            print(f"\n  Evaluating checkpoint-{step}...")
            
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint_path, 
                    num_labels=NUM_LABELS
                )
                
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    tokenizer=tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                    compute_metrics=compute_metrics,
                )

                metrics = trainer.evaluate(tokenized_test_dataset)

                loss = metrics.get('eval_loss')
                accuracy = metrics.get('eval_accuracy')

                new_results.append({
                    "model_folder": folder_name,
                    "checkpoint": step,
                    "test_loss": loss,
                    "test_accuracy": accuracy,
                })
                
                print(f"    -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Error processing checkpoint-{step}: {e}")
        
        # Save results after each folder is processed
        if new_results:
            new_results_df = pd.DataFrame(new_results)
            if not existing_results_df.empty:
                combined_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
            else:
                combined_df = new_results_df
            
            # Remove duplicates (keep the last occurrence in case of re-runs)
            combined_df = combined_df.drop_duplicates(subset=["model_folder", "checkpoint"], keep="last")
            combined_df = combined_df.sort_values(by=["model_folder", "checkpoint"])
            combined_df.to_csv(OUTPUT_RESULTS_FILE, index=False)
            print(f"\n  -> Results saved after processing {folder_name}")
            
            # Update existing_results_df to include new results for next iteration
            existing_results_df = combined_df
    
    # Final summary
    if new_results:
        print(f"\n{'='*60}")
        print(f"All results saved to {OUTPUT_RESULTS_FILE}")
        print(f"Added {len(new_results)} new evaluations")
        print(f"Total evaluations: {len(existing_results_df)}")
        print(f"{'='*60}\n")
        print(existing_results_df.to_string(index=False))
        return existing_results_df
    else:
        print(f"\n{'='*60}")
        print("No new evaluations performed. All target checkpoints already evaluated.")
        print(f"{'='*60}\n")
        if not existing_results_df.empty:
            print(existing_results_df.to_string(index=False))
        return existing_results_df

# Run the evaluation
if __name__ == "__main__":
    results_df = evaluate_all_model_folders()
    
    # Optional: Create plots per model folder
    if not results_df.empty:
        for folder_name in results_df["model_folder"].unique():
            folder_data = results_df[results_df["model_folder"] == folder_name]
            
            output_plots_dir = os.path.join(BASE_DIR, folder_name, "evaluation_plots")
            os.makedirs(output_plots_dir, exist_ok=True)
            
            # Loss plot
            plt.figure(figsize=(12, 6))
            plt.plot(folder_data["checkpoint"], folder_data["test_loss"], marker='o', linestyle='-', color='tab:blue')
            plt.title(f'Test Loss vs. Checkpoint Step ({folder_name})')
            plt.xlabel('Checkpoint Step Number')
            plt.ylabel('Test Loss')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(output_plots_dir, "test_loss_vs_step.png"))
            plt.close()
            
            # Accuracy plot
            plt.figure(figsize=(12, 6))
            plt.plot(folder_data["checkpoint"], folder_data["test_accuracy"], marker='o', linestyle='-', color='tab:green')
            plt.title(f'Test Accuracy vs. Checkpoint Step ({folder_name})')
            plt.xlabel('Checkpoint Step Number')
            plt.ylabel('Test Accuracy')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(output_plots_dir, "test_accuracy_vs_step.png"))
            plt.close()
            
        print(f"\nPlots saved for each model folder in their respective 'evaluation_plots' subdirectories.")