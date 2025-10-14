import torch
from utils.data import load_ag_news
import os
from transformers import AutoModelForSequenceClassification
import os
import torch
from transformers import AutoTokenizer
from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap
import torch
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama

from lxt.efficient import monkey_patch

def get_relevance(model, rel_embeds, ref_embeds, device):
    rel_embeds.requires_grad_(True)
    rel_features = model(inputs_embeds=rel_embeds.requires_grad_(), use_cache=False)[0]
    rel_features = rel_features[0, -1]
    with torch.no_grad():
        ref_features = model(inputs_embeds=ref_embeds, use_cache=False)[0]
        ref_features = ref_features[0, -1]
    output=torch.dot(rel_features,ref_features)
    output.backward()
    relevance = (rel_embeds.grad * rel_embeds).float().sum(-1).detach().cpu()[0]
    relevance = relevance / relevance.abs().max()
    rel_embeds.grad.zero_()
    return relevance

def clean_tokens(words):
    """
    Clean wordpiece tokens by removing special characters and splitting them into words.
    """

    if any("▁" in word for word in words):
        words = [word.replace("▁", " ") for word in words]
    
    elif any("Ġ" in word for word in words):
        words = [word.replace("Ġ", " ") for word in words]
    
    elif any("##" in word for word in words):
        words = [word.replace("##", "") if "##" in word else " " + word for word in words]
        words[0] = words[0].strip()

    else:
        raise ValueError("The tokenization scheme is not recognized.")
    
    special_characters = ['&', '%', '$', '#', '_', '{', '}', '\\']
    for i, word in enumerate(words):
        for special_character in special_characters:
            if special_character in word:
                words[i] = word.replace(special_character, '\\' + special_character)

    return words

def GPTXDA(
        device,
        save_dir,
        test_id,
        train_id,
        loc,
        hf_id,
        tokenizer_hf_id,
        variant=None,
    ):
    # (explainer_class, kwargs)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # train, test = load_tweet_sentiment_dataset(device)
    train, test = load_ag_news()

    # model = GPT2Wrapper(hf_id="herrerovir/gpt2-tweet-sentiment-model", device=device)
    # monkey_patch(modeling_gpt2, verbose=True)
    monkey_patch(modeling_llama, verbose=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id).model

    # elif dataset_name=="ag_news":
        # model = GPT2Wrapper(hf_id="MoritzWeckbecker/gpt2-large_ag-news_full",device="cuda")
    # print_model(model)
    model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_hf_id)
    # tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-large')

    #prompt = """This is awesome!!"""
    #input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    x_train, _ = train[train_id]
    train_input_ids=x_train[0].to(device)
    attention_mask=x_train[1].to(device)
    train_input_ids=train_input_ids[-attention_mask.sum()+1:].unsqueeze(0)
    train_input_embeds = model.get_input_embeddings()(train_input_ids)

    x_test,_ = test[test_id]
    test_input_ids=x_test[0].to(device)
    attention_mask=x_test[1].to(device)
    test_input_ids=test_input_ids[-attention_mask.sum()+1:].unsqueeze(0)
    test_input_embeds = model.get_input_embeddings()(test_input_ids)     

    
    for kw in ["train", "test"]:
        relevance = get_relevance(model,
                      test_input_embeds if kw=="test" else train_input_embeds,
                      train_input_embeds if kw=="test" else test_input_embeds,
                      device)
        tokens = tokenizer.convert_ids_to_tokens(
            test_input_ids[0] if kw=="test" else train_input_ids[0]
            )
        tokens = clean_tokens(tokens)
        os.makedirs(os.path.join(save_dir,kw),exist_ok=True)
        save_path= os.path.join(save_dir,kw,f'attr_order:{loc}-test:{test_id}-train:{train_id}-{kw}_heatmap.pdf')
        pdf_heatmap(tokens, relevance, path=save_path, backend='pdflatex')



# if __name__ == "__main__":
#     # current = os.path.dirname(os.path.realpath(__file__))
#     # parent_directory = os.path.dirname(current)
#     # sys.path.append(current)
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--test_id', type=int)
#     parser.add_argument('--train_id', type=int)
#     parser.add_argument('--save_dir', type=str, default="../text_attributions/tweet_sentiment_extraction/dualda_0.001/sample_0")
#     parser.add_argument('--variant', type=str, default="attn")
#     parser.add_argument('--hf_id', type=str)
#     parser.add_argument('--tokenizer_hf_id', type=str)
#     # parser.add_argument('--page', type=int, default=0)

#     # parser.add_argument('--cache_dir', type=str)
#     # parser.add_argument('--grad_dir', type=str)
#     # parser.add_argument('--features_dir', type=str)

    
#     args = parser.parse_args()

#     print(f"IS CUDA AVAILABLE?: {torch.cuda.is_available()}")
#     GPTXDA(
#                   test_id=args.test_id,
#                   train_id=args.train_id,
#                   save_dir=args.save_dir,
#                   variant=args.variant,
#                   hf_id=args.hf_id,
#                   tokenizer_hf_id=args.tokenizer_hf_id,
#                   device="cuda",
#         )