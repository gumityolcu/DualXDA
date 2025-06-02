import argparse
from utils.data import load_datasets, ReduceLabelDataset, PredictionTargetDataset
from utils.models import clear_resnet_from_checkpoints, load_model
from explain import load_explainer


import yaml
import logging
from metrics import *



def load_metric(metric_name, dataset_name, train, test, device, model, model_name,
                epochs, loss, lr, momentum, optimizer, scheduler,
                weight_decay, augmentation, sample_nr, cache_dir,lds_cache_dir, num_classes, batch_size):
    base_dict={
        "train": train,
        "test": test,
        "device":device
    }
    retrain_dict={
        "dataset_name": dataset_name,
        "model_name": model_name,
        "epochs": epochs,
        "loss": loss,
        "lr": lr,
        "momentum": momentum,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "augmentation": augmentation,
        "num_classes": num_classes,
        "batch_size": batch_size
    }

    """
     dataset_name,  model, 
                 epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, batch_nr=10, mode="cum"
                 
                 
    """
    # dataset_sizes={
    #     "MNIST": 60000,
    #     "CIFAR": 50000,
    #     "AWA": 30000
    # }


    ret_dict = {
                "stdk": (TopKSameClassMetric,{}), "groupk": (TopKSameSubclassMetric,{}),
                "class_detection": (QuandaClassDetection,{"model":model}), "subclass_detection": (QuandaSubclassDetection,{"model":model}), 
                "mark": (MarkImageMetric, {"model":model, "topk": 10}),
                "shortcut_detection":(QuandaShortcutDetection,{"model":model}),
                "corrupt": (CorruptLabelMetric,{}),
                "lds_cache": (LinearDatamodelingScoreCacher, { **retrain_dict, **{'sample_nr': sample_nr, 'cache_dir': cache_dir}}),
                "lindatmod": (LDS, {"model":model, "cache_dir":lds_cache_dir, "pretrained_models": [f"lds0.5_{i:02d}" for i in range(100)]}),
                "add_batch_in": (BatchRetraining,{ **retrain_dict,**{"mode":"cum"}}),
                "add_batch_in_neg": (BatchRetraining, { **retrain_dict,**{"mode":"neg_cum"}}), 
                }
    if metric_name not in ret_dict.keys():
        raise Exception(f"{metric_name} is not a metric name")
    metric_cls, metric_kwargs = ret_dict[metric_name]
    base_dict.update(metric_kwargs)
    return metric_cls(**base_dict)


def evaluate(model_name, model_path, device, class_groups,
             dataset_name, metric_name,
             data_root, xpl_root,
             save_dir, validation_size, num_classes,
             epochs, loss, lr, momentum, optimizer, scheduler,
             weight_decay, augmentation, sample_nr, xai_method, cache_dir, lds_cache_dir, grad_dir, features_dir, batch_size):
    if not torch.cuda.is_available():
        device="cpu"
    if augmentation not in [None, '']:
        augmentation = load_augmentation(augmentation, dataset_name)
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "test",
        'validation_size': validation_size,
        'only_train': False,
        'transform': augmentation,
        'num_classes': num_classes
    }
    if metric_name == "shortcut_detection":
        dataset_type="mark"
    elif metric_name == "subclass_detection":
        dataset_type="group"
    elif metric_name == "mislabeling_detection":
        dataset_type="corrupt"
    else:
        dataset_type = "std"
    train, test = load_datasets(dataset_name, dataset_type, **ds_kwargs)
    if dataset_type=="subclass_detection":
        num_classes_model=len(class_groups)
    else: 
        num_classes_model = num_classes
    model = load_model(model_name, dataset_name, num_classes_model).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint=clear_resnet_from_checkpoints(checkpoint)

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    test=PredictionTargetDataset(dataset=test, model=model, device=device)
    metric = load_metric(metric_name, dataset_name, train, test, device, model, model_name,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, sample_nr, cache_dir, lds_cache_dir, num_classes, batch_size)
    print(f"Computing metric {metric.name}")

    if metric_name == 'lds_cache':
        return

    splitted=xpl_root.split('/')
    if splitted[-1]=="":
        splitted=splitted[:-1]
    outfile_name=splitted[-1]


    if metric_name == "mislabeling_detection":
        explainer_cls, kwargs = load_explainer(xai_method, model_path, save_dir, cache_dir, grad_dir, features_dir, dataset_name, dataset_type)
        train = ReduceLabelDataset(train)
        explainer = explainer_cls(model=model, dataset=train, device=device, **kwargs)
        explainer.train()
        selfinf=explainer.self_influences()
        metric(selfinf)
        if "dualview" in xai_method or "representer" in xai_method:
            selfinf=explainer.self_influences(only_coefs=True)
        else:
            selfinf=None

        metric.get_result(save_dir, f"{dataset_name}_{metric_name}_{outfile_name}_eval_results.json", selfinf)
        return
    
    ################

    
    #check if merged xpl exists
    if not os.path.isdir(xpl_root):
        raise Exception(f"Can not find standard explanation directory {xpl_root}")
    file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f) and (".shark" not in f) and (".times" not in f)]
    file_root = file_list[0].split('_')[0]
    num_files=len(file_list)
    if os.path.isfile(os.path.join(xpl_root, f"{file_root}_all")):
        xpl_all = torch.load(os.path.join(xpl_root, f"{file_root}_all"), map_location=device)
    #merge all xpl
    else:
        xpl_all = torch.empty(0, device=device)
        for i in range(num_files):
            fname = os.path.join(xpl_root, f"{file_root}_{i:02d}")
            xpl = torch.load(fname, map_location=torch.device(device))
            xpl.to(device)
            xpl_all = torch.cat((xpl_all, xpl), 0)
        torch.save(xpl_all, os.path.join(xpl_root, f"{file_root}_all"))
    metric(xpl_all, 0)
    metric.get_result(save_dir, f"{dataset_name}_{metric_name}_{outfile_name}_eval_results.json")


if __name__ == "__main__":
    # current = os.path.dirname(os.path.realpath(__file__))
    # parent_directory = os.path.dirname(current)
    # sys.path.append(current+)
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

    evaluate(
                model_name=train_config.get('model_name', None),
                model_path=train_config.get('model_path', None),
                device=train_config.get('device', 'cuda'),
                class_groups=train_config.get('class_groups', None),
                dataset_name=train_config.get('dataset_name', None),
                metric_name=train_config.get('metric', 'std'),
                data_root=train_config.get('data_root', None),
                xpl_root=train_config.get('xpl_root', None),
                save_dir=train_config.get('save_dir', None),
                validation_size=train_config.get('validation_size', 2000),
                num_classes=train_config.get('num_classes'),
                epochs=train_config.get('epochs', None),
                loss=train_config.get('loss', None),
                lr=train_config.get('lr', None),
                momentum=train_config.get('momentum', None),
                optimizer=train_config.get('optimizer', None),
                scheduler=train_config.get('scheduler', None),
                weight_decay=train_config.get('weight_decay', None),
                augmentation=train_config.get('augmentation', None),
                sample_nr=train_config.get('sample_nr', None),
                xai_method=train_config.get('xai_method', None),
                cache_dir=train_config.get('cache_dir', None),
                lds_cache_dir=train_config.get('lds_cache_dir', None),
                grad_dir=train_config.get('grad_dir', None),
                features_dir=train_config.get('features_dir', None),
                batch_size=train_config.get('batch_size', 64)
    )