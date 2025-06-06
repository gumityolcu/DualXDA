import argparse
import torch
from utils import xplain
from utils.explainers import GradCosExplainer, GradDotExplainer
from explainers import TRAK, DualView, RepresenterPointsExplainer, LiSSAInfluenceFunctionExplainer, TracInExplainer, ArnoldiInfluenceFunctionExplainer, KronfluenceExplainer, FeatureSimilarityExplainer, InputSimilarityExplainer
from utils.data import load_datasets_reduced
from utils.models import clear_resnet_from_checkpoints, compute_accuracy, load_model
from time import time
import logging
import os

def count_params(checkpoint):
    total=0
    for p,v in checkpoint["model_state"].items():
        num=1
        for s in v.data.shape:
            num=num*s
        total=total+num
    return total

def count_params_model(model):
    total=0
    for v in model.parameters():
        num=1
        for s in v.data.shape:
            num=num*s
        total=total+num
    return total

def load_explainer(xai_method, model_path, save_dir, cache_dir, grad_dir, features_dir, dataset_name, dataset_type):
    lissa_params={
        "MNIST": {'depth': 6000, 'repeat': 10, "file_size": 20},
        "CIFAR": {'depth': 5000, 'repeat': 10, "file_size":20},
        "AWA": {'depth': 3700, 'repeat': 10, "file_size":20}
    }

    arnoldi_params={
        "MNIST": {"projection_dim": 128, "arnoldi_dim":150, "hessian_dataset_size": 10000},
        "CIFAR": {"projection_dim": 128, "arnoldi_dim":150, "hessian_dataset_size": 10000},
        "AWA": {"projection_dim": 128, "arnoldi_dim": 150, "hessian_dataset_size": 10000},
    }
    kronfluence_params={
         "score_data_partitions":20 if dataset_name=="CIFAR" and dataset_type=="mark" else 10
    }

    # if we want seperate kwargs for each dataset, below is a dictionary with default values
    # kronfluence_params={
    #     "covariance_max_examples": 100000,
    #     "covariance_data_partitions": 1,
    #     "lambda_max_examples": 100000,
    #     "lambda_data_partitions": 1,
    #     "use_iterative_lambda_aggregation": False,
    #     "score_data_partitions":1
    #     }

    trak_params={
        "MNIST": {'proj_dim': 2048, "base_cache_dir":cache_dir, "dir": save_dir},
        "CIFAR": {'proj_dim': 2048, "base_cache_dir":cache_dir, "dir": save_dir},
        "AWA": {'proj_dim': 2048, "base_cache_dir":cache_dir, "dir": save_dir, "batch_size": 1},
    }

    dualview_params={}

    graddot_params={}

    tracin_params={}

    representer_params={}

    explainers = {
        'representer': (RepresenterPointsExplainer, {"dir": cache_dir, "features_dir": features_dir}),
        'trak': (TRAK, trak_params[dataset_name]),# trak writes to the cache during explanation. so we can't share cache between jobs. therefore, each job uses the save_dir to copy the cache and deletes the cache folder from save_dir before quitting the job
        'dualview': (DualView, {"dir": cache_dir, "features_dir":features_dir}),
        'graddot': (GradDotExplainer, {"mat_dir":cache_dir, "grad_dir":grad_dir,  "dimensions":128}),
        #'gradcos': (GradCosExplainer, {"dir":cache_dir, "dimensions":128,  "ds_type": dataset_type}),
        'tracin': (TracInExplainer, {'ckpt_dir':os.path.dirname(model_path), 'dir':cache_dir, 'dimensions':128}),
        'lissa': (LiSSAInfluenceFunctionExplainer, {'dir':cache_dir, 'scale':10, **lissa_params[dataset_name]}),
        'arnoldi': (ArnoldiInfluenceFunctionExplainer, {'dir':cache_dir, 'batch_size':32, 'seed':42, **arnoldi_params[dataset_name]}),
        'kronfluence': (KronfluenceExplainer, {'dir':cache_dir, **kronfluence_params}),
        'feature_similarity_dot': (FeatureSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "dot"}),
        'feature_similarity_cos': (FeatureSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "cos"}),
        'feature_similarity_l2': (FeatureSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "l2"}),
        'input_similarity_dot': (InputSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "dot"}),
        'input_similarity_cos': (InputSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "cos"}),
        'input_similarity_l2': (InputSimilarityExplainer, {'dir':cache_dir, "features_dir": features_dir, "mode": "l2"}),
     }    
    return explainers[xai_method]

def print_model(model):
    total=0
    marginals=[]
    for name, params in model.named_parameters():
        this=0
        num=1
        for s in params.shape:
            num=num*s
        marginals.append((name,num))
        total=total+num
    cum=0
    for name, count in marginals:
        cum=cum+count
        print(name, "Percentage: ", float(count)/float(total), "Cumulative: ", float(cum)/float(total))
    print("TOTAL:",total)

def explain_model(model_name, model_path, device,dataset_name, dataset_type,
                  data_root, batch_size, save_dir, validation_size, testsplit):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num_classes = 50 if dataset_name == "AWA" else 10 
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': None,
        'image_set': "test",
        'validation_size': validation_size,
        "only_train": False,
        'testsplit': testsplit,
        'transform': None,
        'num_classes': num_classes, 
    }

    train, test = load_datasets_reduced(dataset_name, dataset_type, ds_kwargs)
    model = load_model(model_name, dataset_name, num_classes)

    checkpoint = torch.load(model_path, map_location=device)
    #get rid of model.resnet
    checkpoint=clear_resnet_from_checkpoints(checkpoint)

    model.load_state_dict(checkpoint["model_state"])
    # print_model(model)
    # exit()
    model.to(device)
    model.eval()
    loader=torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    times=[]
    for x,y in loader:
        t0=time()
        x=x.to(device)
        y=y.to(device)
        _ = model.features(x)
        times.append((time()-t0)/x.shape[0])
    print(times)
    print("*********************")
    print(f"Average time per sample for {dataset_name}: {sum(times)/len(times)} seconds")


if __name__ == "__main__":
    # current = os.path.dirname(os.path.realpath(__file__))
    # parent_directory = os.path.dirname(current)
    # sys.path.append(current)
    model_names = {
        "MNIST": "basic_conv",
        "CIFAR": "resnet18",
        "AWA": "resnet50"
    }
    for ds_name in model_names.keys():
        model_path=f"/mnt/checkpoints/{ds_name}/{ds_name}_{model_names[ds_name]}_best.pth"
        explain_model(model_name=model_names[ds_name],
                  model_path=model_path,
                  device='cuda',
                  class_groups=None,
                  dataset_name=ds_name,
                  dataset_type="std",
                  data_root="/mnt/data",
                  batch_size=32,
                  save_dir="/mnt/outputs",
                  validation_size=2000,
                  testsplit="test" if ds_name != "AWA" else "val",
                  )