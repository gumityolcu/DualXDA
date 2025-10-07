import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    DataCollatorWithPadding,
)
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
import torch

# --- Configuration ---
MODEL_BASE_DIR = "/home/weckbecker/coding/DualXDA/src/models/gpt2-ag_news/train_full"
TEST_FILE_PATH = "/home/weckbecker/coding/DualXDA/src/dataset/ag_news/test.jsonl"
OUTPUT_PLOTS_DIR = os.path.join(MODEL_BASE_DIR, "evaluation_plots")
OUTPUT_RESULTS_FILE = os.path.join(MODEL_BASE_DIR, "checkpoint_evaluation_results.csv")
MODEL_NAME_OR_PATH = "gpt2" # Use the base model name to load the tokenizer, 
                           # but the checkpoints will load the model weights.
NUM_LABELS = 4 # AG News has 4 classes

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
# 3. & 4. Setup Evaluation and Compute Metrics
# --------------------------

# a) Define the metric function
def compute_metrics(p: EvalPrediction):
    """Computes accuracy for sequence classification."""
    preds = p.predictions[0].argmax(-1)
    labels = p.label_ids
    return {"accuracy": accuracy_score(labels, preds)}

# b) Load Tokenizer and Dataset

# We load the tokenizer from the *last* checkpoint or the base folder, 
# assuming all checkpoints use the same one.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_DIR, use_fast=True)
except:
    print(f"Warning: Could not load tokenizer from {MODEL_BASE_DIR}. Trying base model {MODEL_NAME_OR_PATH}.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)

# GPT-2 does not have a padding token by default, which is needed for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the test dataset from the jsonl file
raw_test_dataset = load_dataset("json", data_files=TEST_FILE_PATH, split="train")

# Preprocessing function for the dataset
def preprocess_function(examples):
    # Adjust for your dataset's column names ('text' and 'label' are common)
    return tokenizer(examples["text"], truncation=True, padding="max_length")

# Apply preprocessing
tokenized_test_dataset = raw_test_dataset.map(preprocess_function, batched=True)

# Select only the columns required by the model and rename 'label' to 'labels'
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])
tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")
tokenized_test_dataset.set_format("torch")


# c) Setup Trainer arguments
# The TrainingArguments are minimal for evaluation-only
training_args = TrainingArguments(
    output_dir="./evaluation_output",  # Temporary directory for Trainer output
    per_device_eval_batch_size=32,
    do_eval=True,
    report_to="none", # Don't log to external services
)

# --------------------------
# 5. Loop Through Checkpoints and Evaluate
# --------------------------

checkpoint_results = []
checkpoints = get_checkpoints(MODEL_BASE_DIR)

if not checkpoints:
    print(f"No checkpoint directories found in {MODEL_BASE_DIR}")
else:
    print(f"Found {len(checkpoints)} checkpoints. Starting evaluation...")
    
    # Get the checkpoint path for