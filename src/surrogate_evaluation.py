# Implement two surrogate metrics:
# 1) How close are the weight vectors (Cosine similarity or Euclidean distance - need to normalise weight vector for eculidean before?)
# 2) How close are the activations under these models? (e.g. correlation as in https://arxiv.org/pdf/2305.14585.pdf, which ignores the scale - good)
# 3) How close are the decisions by these models? (e.g. ratio of mistakes)

# Problems with 2 + 3: Can have the same activations/same decisions for completely different reasons/weight vectors, especially, when neurons are highly correlated
# Problem with 1: Can not differentiate between scale of neurons (e.g. if one neuron is ten times larger than the other, we expect the weight entry to be only the tenth for their product to be on the same scale)

# Hopefully, if both things are close, our surrogate model should be good

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import matthews_corrcoef
import numpy as np
import os
import torch
from utils.data import load_datasets_reduced
from utils.models import load_model, clear_resnet_from_checkpoints
import argparse
import yaml
import logging
import csv
from scipy.stats import kendalltau

from torchmetrics.regression import KendallRankCorrCoef
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from explain import load_explainer

def load_surrogate(model_name, model_path, device,
                     class_groups, dataset_name, dataset_type,
                     data_root, cache_dir, grad_dir, features_dir, batch_size, save_dir_explainer, save_dir_results,
                     validation_size, xai_method,num_classes, C_margin, testsplit
                     ):
    # (explainer_class, kwargs)
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)
    if not torch.cuda.is_available():
        device="cpu"

    ds_kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "test",
        'validation_size': validation_size,
        "only_train": False,
        'testsplit': testsplit,
        'transform': None,
        'num_classes': num_classes
    }



    train, test = load_datasets_reduced(dataset_name, dataset_type, ds_kwargs)
    model = load_model(model_name, dataset_name, num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint=clear_resnet_from_checkpoints(checkpoint)
    
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    explainer_cls, kwargs=load_explainer(xai_method, model_path, save_dir_explainer, cache_dir, grad_dir, features_dir, dataset_name, dataset_type)

    if C_margin is not None:
        if xai_method=="dualview":
            kwargs["C"]=C_margin
        elif xai_method=="representer":
            kwargs["sparsity"]=C_margin
    
    explainer = explainer_cls(model=model, dataset=train, device=device, **kwargs)
    
    if xai_method == "dualview":
        explainer.read_variables()
        w1 = explainer.learned_weight
        w2 = explainer.coefficients.float().T @ explainer.samples
        print("SANITY ", ((w1 - w2)).abs().mean().item())
    else:
        explainer.train()
        w1 = explainer.learned_weight
        w2 = explainer.coefficients.float().T @ explainer.samples
        print("SANITY ", ((w1 - w2)).abs().mean().item())

    print(f"Checking surrogate faithfulness of {explainer_cls.name}")

    model_weights = model.classifier.weight.detach() #and bias?
    surrogate_weights = explainer.learned_weight
    loader=torch.utils.data.DataLoader(train, 32, shuffle=False) #concat train and test and check activations on both
    model_logits=torch.empty((0,num_classes)).to(device)
    for x, y in iter(loader): #tqdm.tqdm(loader)
        x=x.to(device)
        y=y.to(device)
        _model_logits = model(x).detach()
        model_logits=torch.cat((model_logits, _model_logits), dim=0)    
    model_predictions = torch.argmax(model_logits, dim=1)
    model_preactivations = explainer.samples
    surrogate_logits = torch.matmul(model_preactivations, surrogate_weights.T)
    surrogate_predictions = torch.argmax(surrogate_logits, dim=1)

    score_cos_weights = surrogate_faithfulness_cosine(model_weights.to(device), surrogate_weights.to(device))
    score_cos_logits = surrogate_faithfulness_logits(model_logits.to(device), surrogate_logits.to(device))
    score_matthews_predictions = surrogate_faithfulness_prediction(model_predictions.to(device), surrogate_predictions.to(device))
    score_kendall_logits = surrogate_faithfulness_logits_kendall(model_logits.to(device), surrogate_logits.to(device))

    print("\n")
    print("Cosine similarity of weight matrices:", score_cos_weights)
    print("Correlation of logits:", score_cos_logits)
    print("Correlation of prediction:", score_matthews_predictions)
    print("Kendall tau-rank correlation of logits:", score_kendall_logits)
    print("\n")
    
    loader=torch.utils.data.DataLoader(test, 32, shuffle=False) #concat train and test and check activations on both
    model_logits=torch.empty((0,num_classes)).to(device)
    test_classifications=[]
    for x, y in iter(loader): #tqdm.tqdm(loader)
        x=x.to(device)
        y=y.to(device)
        _model_logits = model(x).detach()
        model_logits=torch.cat((model_logits, _model_logits), dim=0)
        preds=torch.argmax(_model_logits, dim=1)
        test_classifications.append(preds==y)
    test_classifications=torch.cat(test_classifications, dim=0)

    results_dict = [{"Metric": "Cosine similarity of weight matrices", "Score": score_cos_weights},
                    {"Metric": "Correlation of logits", "Score": score_cos_logits},
                    {"Metric": "Correlation of prediction", "Score": score_matthews_predictions},
                    {"Metric": "Kendall tau-rank correlation of logits", "Score": score_kendall_logits},]
    
    with open(os.path.join(save_dir_results ,f"{dataset_name}_{dataset_type}_surrogate_evaluation.csv"), "w") as file: 
        writer = csv.DictWriter(file, fieldnames = ['Metric', 'Score'])
        writer.writeheader()
        writer.writerows(results_dict)


def surrogate_faithfulness_cosine(model_weights, surrogate_weights):
    #old_score = np.average(np.diag(cosine_similarity(model_weights.numpy(), surrogate_weights.numpy())))
    score = pairwise_cosine_similarity(model_weights, surrogate_weights).diag().mean().item()
    #print(score, old_score)
    return score

def surrogate_faithfulness_logits(model_logits, surrogate_logits):
    #old_score = np.average(np.diag(cosine_similarity(model_logits.numpy(), surrogate_logits.numpy())))
    score = pairwise_cosine_similarity(model_logits, surrogate_logits).diag().mean().item()
    #print(score, old_score)
    return score
    
def surrogate_faithfulness_prediction(model_predictions, surrogate_predictions):
    matthews = MulticlassMatthewsCorrCoef(num_classes = 1 + int(torch.max(model_predictions.max(), surrogate_predictions.max()))).to(surrogate_predictions.device)
    #old_score = matthews_corrcoef(model_predictions.numpy(), surrogate_predictions.numpy())
    score = matthews(model_predictions, surrogate_predictions).item()
    #print(score, old_score)
    return score

#def surrogate_test_accuracy()

# kendall tau-rank used in "Faithful and Efficient Explanations for NNs via Neural Tangent Kernel Surrogate Models"
def surrogate_faithfulness_logits_kendall(model_logits, surrogate_logits):
    kendall = KendallRankCorrCoef(num_outputs=model_logits.shape[0]).to(model_logits.device)
    #old_score = np.average([kendalltau(model_logits[i,:].argsort(descending=True).numpy(), surrogate_logits[i,:].argsort(descending=True).numpy()).statistic for i in range(len(model_logits))])
    #print(model_logits.shape)
    #print(surrogate_logits.shape)
    score = kendall(model_logits.T, surrogate_logits.T).mean().item()
    #print(score, old_score)
    return score

# talk to Galip how to get weight vector and bias vector from surrogate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            surrogate_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    load_surrogate(model_name=surrogate_config.get('model_name', None),
                     model_path=surrogate_config.get('model_path', None),
                     device=surrogate_config.get('device', 'cuda'),
                     class_groups=surrogate_config.get('class_groups', None),
                     dataset_name=surrogate_config.get('dataset_name', None),
                     dataset_type=surrogate_config.get('dataset_type', 'std'),
                     data_root=surrogate_config.get('data_root', None),
                     cache_dir=surrogate_config.get("cache_dir", None),
                     grad_dir=surrogate_config.get("grad_dir",None),
                     features_dir=surrogate_config.get("features_dir",None),
                     batch_size=surrogate_config.get('batch_size', None),
                     save_dir_explainer=surrogate_config.get('save_dir_explainer', None),
                     save_dir_results=surrogate_config.get('save_dir_results', None),
                     validation_size=surrogate_config.get('validation_size', 2000),
                     xai_method=surrogate_config.get('xai_method', None),
                     num_classes=surrogate_config.get('num_classes'),
                     C_margin=surrogate_config.get('C',None),
                     testsplit=surrogate_config.get('testsplit',"test")
                     )
