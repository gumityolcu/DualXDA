import argparse
import torch
from utils import xplain
from utils.explainers import GradCosExplainer, GradDotExplainer
from explainers import TRAK, DualView, RepresenterPointsExplainer, InfluenceFunctionExplainer, TracInExplainer
from utils.data import load_datasets_reduced
from utils.models import compute_accuracy, load_model
import yaml
import logging
import os


def load_explainer(xai_method, model_path, save_dir, dataset_name):
    if_params={
        "MNIST": {'depth': 50, 'repeat': 1200},
        "CIFAR": {'depth': 50, 'repeat': 1000},
        "AWA": {'depth': 50, 'repeat': 500}
    }
    explainers = {
        'representer': (RepresenterPointsExplainer, {"dir": save_dir}),
        'trak': (TRAK, {'proj_dim': 512, "dir":save_dir}),
        'dualview': (DualView, {"dir": save_dir}),
        'graddot': (GradDotExplainer, {"dir":save_dir, "dimensions":128}),
        'gradcos': (GradCosExplainer, {"dir":save_dir, "dimensions":128}),
        'tracin': (TracInExplainer, {'ckpt_dir':os.path.dirname(model_path), 'dir':save_dir, 'dimensions':1}),
        'influence': (InfluenceFunctionExplainer, if_params[dataset_name])
    }
    return explainers[xai_method]


def explain_model(model_name, model_path, device, class_groups,
                  dataset_name, dataset_type, data_root, batch_size,
                  save_dir, validation_size, num_batches_per_file,
                  start_file, num_files, xai_method,
                  num_classes, C_margin, testsplit):
    # (explainer_class, kwargs)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not torch.cuda.is_available():
        device = "cpu"
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
    model = load_model(model_name, dataset_name, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    # if accuracy:
    #    acc, err = compute_accuracy(model, test,device)
    #    print(f"Accuracy: {acc}")
    explainer_cls, kwargs = load_explainer(xai_method, model_path, save_dir, dataset_name)
    if C_margin is not None:
        kwargs["C"] = C_margin
    print(f"Generating explanations with {explainer_cls.name}")
    xplain(
        model=model,
        train=train,
        test=test,
        device=device,
        explainer_cls=explainer_cls,
        kwargs=kwargs,
        batch_size=batch_size,
        num_batches_per_file=num_batches_per_file,
        save_dir=save_dir,
        start_file=start_file,
        num_files=num_files
    )


if __name__ == "__main__":
    # current = os.path.dirname(os.path.realpath(__file__))
    # parent_directory = os.path.dirname(current)
    # sys.path.append(current)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    save_dir = f"{train_config['save_dir']}/{os.path.basename(config_file)[:-5]}"

    explain_model(model_name=train_config.get('model_name', None),
                  model_path=train_config.get('model_path', None),
                  device=train_config.get('device', 'cuda'),
                  class_groups=train_config.get('class_groups', None),
                  dataset_name=train_config.get('dataset_name', None),
                  dataset_type=train_config.get('dataset_type', 'std'),
                  data_root=train_config.get('data_root', None),
                  batch_size=train_config.get('batch_size', None),
                  save_dir=train_config.get('save_dir', None),
                  validation_size=train_config.get('validation_size', 2000),
                  num_batches_per_file=train_config.get('num_batches_per_file', 10),
                  start_file=train_config.get('start_file', 0),
                  num_files=train_config.get('num_files', 100),
                  xai_method=train_config.get('xai_method', None),
                  num_classes=train_config.get('num_classes'),
                  C_margin=train_config.get('C', None),
                  testsplit=train_config.get('testsplit', "test"),
                  )
