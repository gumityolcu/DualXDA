import argparse
import torch
from utils.data import load_tweet_sentiment_dataset, load_ag_news
from utils.models import LlamaWrapper
import os
from GPTXDA import GPTXDA

# def count_params(checkpoint):
#     total=0
#     for p,v in checkpoint["model_state"].items():
#         num=1
#         for s in v.data.shape:
#             num=num*s
#         total=total+num
#     return total

# def count_params_model(model):
#     total=0
#     for v in model.parameters():
#         num=1
#         for s in v.data.shape:
#             num=num*s
#         total=total+num
#     return total

# def load_explainer(xai_method, model_path, save_dir, cache_dir, grad_dir, features_dir, dataset_name, dataset_type, sparse):
#     lissa_params={
#         "MNIST": {'depth': 6000, 'repeat': 10, "file_size": 20},
#         "CIFAR": {'depth': 5000, 'repeat': 10, "file_size":20},
#         "AWA": {'depth': 3700, 'repeat': 10, "file_size":20},
#         "tweet_sentiment_extraction": {'depth': 3700, 'repeat': 10, "file_size":20}
#     }

#     arnoldi_params={
#         "MNIST": {"projection_dim": 128, "arnoldi_dim":150, "hessian_dataset_size": 10000},
#         "CIFAR": {"projection_dim": 128, "arnoldi_dim":150, "hessian_dataset_size": 10000},
#         "AWA": {"projection_dim": 128, "arnoldi_dim": 150, "hessian_dataset_size": 10000},
#         # "tweet_sentiment_extraction": {"projection_dim": 256, "arnoldi_dim": 500, "hessian_dataset_size": 1500},
#         "tweet_sentiment_extraction": {"projection_dim": 2, "arnoldi_dim": 2, "hessian_dataset_size": 2},
#     }

#     kronfluence_params={
#          "score_data_partitions":20 if dataset_name=="CIFAR" and dataset_type=="mark" else 10
#     }

#     # if we want seperate kwargs for each dataset, below is a dictionary with default values
#     # kronfluence_params={
#     #     "covariance_max_examples": 100000,
#     #     "covariance_data_partitions": 1,
#     #     "lambda_max_examples": 100000,
#     #     "lambda_data_partitions": 1,
#     #     "use_iterative_lambda_aggregation": False,
#     #     "score_data_partitions":1
#     #     }

#     trak_params={
#         "MNIST": {'proj_dim': 2048, "base_cache_dir":cache_dir, "dir": save_dir},
#         "CIFAR": {'proj_dim': 2048, "base_cache_dir":cache_dir, "dir": save_dir},
#         "AWA": {'proj_dim': 2048, "base_cache_dir":cache_dir, "dir": save_dir, "batch_size": 1},
#         "tweet_sentiment_extraction": {'proj_dim': 4096, "base_cache_dir":cache_dir, "dir": save_dir, "batch_size": 1}
#     }

#     dualda_params={}

#     graddot_params={}

#     tracin_params={}

#     representer_params={}

#     explainers = {
#         'representer': (RepresenterPointsExplainer, {"dir": cache_dir, "features_dir": features_dir}),
#         'trak': (TRAK, trak_params[dataset_name]),# trak writes to the cache during explanation. so we can't share cache between jobs. therefore, each job uses the save_dir to copy the cache and deletes the cache folder from save_dir before quitting the job
#         'dualda': (DualDA, {"dir": cache_dir, "features_dir":features_dir}),
#         'graddot': (GradDotExplainer, {"mat_dir":cache_dir, "grad_dir":grad_dir,  "dimensions":128}),
#         #'gradcos': (GradCosExplainer, {"dir":cache_dir, "dimensions":128,  "ds_type": dataset_type}),
#         'tracin': (TracInExplainer, {'ckpt_dir':os.path.dirname(model_path), 'dir':cache_dir, 'dimensions':128}),
#         'lissa': (LiSSAInfluenceFunctionExplainer, {'dir':cache_dir, 'scale':10, **lissa_params[dataset_name]}),
#         'arnoldi': (ArnoldiInfluenceFunctionExplainer, {'dir':cache_dir, 'batch_size':32, 'seed':42, **arnoldi_params[dataset_name]}),
#         'kronfluence': (KronfluenceExplainer, {'dir':cache_dir, **kronfluence_params}),
#         'feature_similarity_dot': (FeatureSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "dot"}),
#         'feature_similarity_cos': (FeatureSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "cos"}),
#         'feature_similarity_l2': (FeatureSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "l2"}),
#         'input_similarity_dot': (InputSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "dot"}),
#         'input_similarity_cos': (InputSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "cos"}),
#         'input_similarity_l2': (InputSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "l2"}),
#      }    
#     return explainers[xai_method]

# def print_model(model):
#     total=0
#     marginals=[]
#     for name, params in model.named_parameters():
#         this=0
#         num=1
#         for s in params.shape:
#             num=num*s
#         marginals.append((name,num))
#         total=total+num
#     cum=0
#     for name, count in marginals:
#         cum=cum+count
#         print(name, "Percentage: ", float(count)/float(total), "Cumulative: ", float(cum)/float(total))
#     print("TOTAL:",total)


def text_attributions(
                  device,
                  dataset_name,
                  xai_method,
                  save_dir,
                  start,
                  length
                  ):
    hf_ids={
        "ag_news": "MoritzWeckbecker/Llama-3.2-1B_ag-news-0",
        "tweet_sentiment_extraction":"herrerovir/gpt2-tweet-sentiment-model"
    }
    tokenizer_hf_ids={
        "ag_news": "unsloth/Llama-3.2-1B",
        "tweet_sentiment_extraction": "herrerovir/gpt2-tweet-sentiment-model"
    }
    # (explainer_class, kwargs)
    if not torch.cuda.is_available():
        device = "cpu"
    if dataset_name=="tweet_sentiment_extraction":
        train, test = load_tweet_sentiment_dataset(device)
    elif dataset_name=="ag_news":
        train, test = load_ag_news()
    model = LlamaWrapper(hf_id=hf_ids[dataset_name],device="cuda")
    # print_model(model)
    model.to(device)
    model.eval()
    
    xpl_root=f"../explanations/{dataset_name}/std/{xai_method}"
    files=[f for f in os.listdir(xpl_root) if not f.endswith(".times") and not f.endswith("_all")]
    
    base_name=os.listdir(xpl_root)[0].split("_")[0]

    if os.path.isfile(os.path.join(xpl_root, f"{base_name}_all")):
        xpl_all = torch.load(os.path.join(xpl_root, f"{base_name}_all"), map_location=device)
    #merge all xpl
    else:
        xpl_all = torch.empty(0, device=device)
        for i in range(len(files)):
            fname = os.path.join(xpl_root, f"{base_name}_{i:02d}")
            xpl = torch.load(fname, map_location=torch.device(device))
            xpl.to(device)
            xpl_all = torch.cat((xpl_all, xpl), 0)
        torch.save(xpl_all, os.path.join(xpl_root, f"{base_name}_all"))

    xpl=torch.load(f"../explanations/{dataset_name}/std/{xai_method}/{base_name}_all")
    
    for i in range(length):
        ret_str=""
        x,y = test[start+i]
        x=x.to(device)
        y=y.to(device)
        test_label=test.label_text[model(x[None]).argmax()]
        ret_str=ret_str+f"TEST SAMPLE-{start+i+1} ({test_label}): \n"+test.get_string(start+i)+"\n\n"
        high=xpl[start+i].argsort(descending=True)[:5]
        low=xpl[start+i].argsort()[:5]
        ret_str=ret_str+"POSITIVE ATTRIBUTIONS\n"
        for j in range(5):
            _,y = train[high[j]]
            ret_str=ret_str+f"Positive-{j+1} ({train.label_text[y]}, {xpl[start+i,high[j]]:.2f}): "+train.get_string(high[j])+"\n\n"
            GPTXDA(
                device=device,
                save_dir=os.path.join(save_dir,xai_method,str(i+start),"POSITIVE"),
                test_id=start+i,
                train_id=high[j],
                hf_id=hf_ids[dataset_name],
                tokenizer_hf_id=tokenizer_hf_ids[dataset_name],
            )
        ret_str=ret_str+"NEGATIVE ATTRIBUTIONS\n"
        for j in range(5):
            _,y = train[low[j]]
            ret_str=ret_str+f"Negative-{j+1} ({train.label_text[y]}, {xpl[start+i,low[j]]:.2f}): "+train.get_string(low[j])+"\n\n"
            GPTXDA(
                device=device,
                save_dir=os.path.join(save_dir,xai_method,str(i+start),"NEGATIVE"),
                test_id=start+i,
                train_id=high[j],
                hf_id=hf_ids[dataset_name],
                tokenizer_hf_id=tokenizer_hf_ids[dataset_name]
            )
        print(ret_str)
        if save_dir is not None:
            os.makedirs(os.path.join(save_dir,xai_method),exist_ok=True)
            with open(os.path.join(save_dir,xai_method,str(i+start),f"attributions"),"w") as f:
                f.write(ret_str)


if __name__ == "__main__":
    # current = os.path.dirname(os.path.realpath(__file__))
    # parent_directory = os.path.dirname(current)
    # sys.path.append(current)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--dataset_name', type=str, default="ag_news")
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--xai_method', type=str)
    parser.add_argument('--length', type=int, default=20)
    parser.add_argument('--start', type=int, default=0)
    # parser.add_argument('--page', type=int, default=0)

    # parser.add_argument('--cache_dir', type=str)
    # parser.add_argument('--grad_dir', type=str)
    # parser.add_argument('--features_dir', type=str)

    
    args = parser.parse_args()

    print(f"IS CUDA AVAILABLE?: {torch.cuda.is_available()}")
    text_attributions(
                  device=args.device,
                  dataset_name=args.dataset_name,
                  xai_method=args.xai_method,
                #   page=args.page,
                  start=args.start,
                  length=args.length,
                  save_dir=args.save_dir,
                  )