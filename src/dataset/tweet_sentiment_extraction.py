import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class TweetSentimentDatapoint(dict):
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self["input_ids"] = input_ids
        self["attention_mask"] = attention_mask
    
    def to(self, device):
        self["input_ids"] = self["input_ids"].to(device)
        self["attention_mask"] = self["attention_mask"].to(device)
        return self
    
    def size(self, i):
        return self["input_ids"].size(i)
    
    @property
    def shape(self):
        return self['input_ids'].shape

class TweetSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, split, device):
        dataset = load_dataset("mteb/tweet_sentiment_extraction")
        self.tokenizer = AutoTokenizer.from_pretrained("herrerovir/gpt2-tweet-sentiment-model")
        # Preprocess
        def preprocess(batch):
            return self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        self.device=device
        self.encoded = dataset.map(preprocess, batched=True)
        self.encoded = self.encoded.rename_column("label", "labels")
        self.label_text=["negative", "neutral", "positive"]
        self.encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        self.split = split
    
    def get_string(self, id, skip_special_tokens=True):
        id=int(id)
        x = self.encoded[self.split][id]
        # mask=x["attention_mask"]
        # final_id=mask.sum()-1
        x=x["input_ids"]
        string=self.tokenizer.batch_decode(x,skip_special_tokens=skip_special_tokens)
        return "".join(string)

    def __len__(self):
        return len(self.encoded[self.split])

    def __getitem__(self, idx):
        idx=int(idx)
        input_ids=self.encoded[self.split][idx]["input_ids"]
        attention_mask=self.encoded[self.split][idx]["attention_mask"]
        return torch.stack((input_ids, attention_mask), dim=0).to(self.device), self.encoded[self.split][idx]["labels"].to(self.device)
        # return TweetSentimentDatapoint(**{key: self.encoded[self.split][idx][key].to(self.device) for key in ["input_ids", "attention_mask"]}), self.encoded[self.split][idx]["labels"].to(self.device)
