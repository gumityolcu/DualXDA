from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.data import load_datasets_reduced, load_tweet_sentiment_dataset, load_ag_news
from utils.models import GPT2Wrapper

# Load dataset
# dataset = load_dataset("mteb/tweet_sentiment_extraction")

# Load model/tokenizer


model = GPT2Wrapper(hf_id="MoritzWeckbecker/gpt2-large_ag-news_full",device="cuda")
model.to("cuda")
model.eval()
train, test = load_ag_news()


# Simple accuracy function
def compute_accuracy(ds):
    dataloader = torch.utils.data.DataLoader(dataset=ds, batch_size=32)
    correct, total = 0, 0
    for x,y in dataloader:
        x=x.to("cuda")
        y=y.to('cuda')
        with torch.no_grad():
            outputs = model(x)
            preds = torch.argmax(outputs, dim=-1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total

train_acc = compute_accuracy(train)
test_acc = compute_accuracy(test)

print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)
