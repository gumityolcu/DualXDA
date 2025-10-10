from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.data import load_datasets_reduced, load_tweet_sentiment_dataset, load_ag_news
from utils.models import GPT2Wrapper, LlamaWrapper

# Load dataset
dataset = load_dataset("mteb/tweet_sentiment_extraction")

# Load model/tokenizer

# orig_model=AutoModelForSequenceClassification.from_pretrained("MoritzWeckbecker/Llama-3.2-1B_ag-news-0")
tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Llama-3.2-1B",
        max_length=1024,
    )
    # Use existing EOS token as pad token (no new token added)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# Now set it in the model config
# orig_model.config.pad_token_id = tokenizer.pad_token_id
# orig_model.to("cpu")
# orig_model.eval()
model = LlamaWrapper(hf_id="MoritzWeckbecker/Llama-3.2-1B_ag-news-0",device="cpu")
model.to("cuda")
model.eval()
train, test = load_ag_news()

# Simple accuracy function
def compute_accuracy(ds):
    shapes=[]
    dataloader = torch.utils.data.DataLoader(dataset=ds, batch_size=32)
    correct, total = 0, 0
    for x,y in dataloader:
        x=x.to("cuda")
        y=y.to('cuda')
        shapes.append(x.shape[2])
        with torch.no_grad():
            preds = torch.argmax(model(x), dim=-1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total

train_acc = compute_accuracy(train)
test_acc = compute_accuracy(test)

print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)
