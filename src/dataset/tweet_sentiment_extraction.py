import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class TweetSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, split, device):
        dataset = load_dataset("mteb/tweet_sentiment_extraction")
        tokenizer = AutoTokenizer.from_pretrained("herrerovir/gpt2-tweet-sentiment-model")
        # Preprocess
        def preprocess(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        self.device=device
        self.encoded = dataset.map(preprocess, batched=True)
        self.encoded = self.encoded.rename_column("label", "labels")
        self.encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        self.split = split
    
    def __len__(self):
        return len(self.encoded[self.split])

    def __getitem__(self, idx):
        return {key: self.encoded[self.split][idx][key].to(self.device) for key in ["input_ids", "attention_mask"]}, self.encoded[self.split][idx]["labels"].to(self.device)
