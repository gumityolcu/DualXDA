import argparse
from utils.data import load_datasets_reduced
from utils.models import load_model
import yaml
import logging
from metrics import *


def load_metric(metric_name, dataset_name, train, test, device, coef_root, model, model_name,
                epochs, loss, lr, momentum, optimizer, scheduler,
                weight_decay, augmentation,):
    base_dict={
        "train":train,
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
        "augmentation": augmentation
    }

    """
     dataset_name,  model, 
                 epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, batch_nr=10, mode="cum"
                 
                 
    """

    ret_dict = {"std": (SameClassMetric,{}), "group": (SameSubclassMetric,{}), 
                "corrupt": (CorruptLabelMetric,{"coef_root": coef_root}),
                "mark": (MarkImageMetric, {"model":model}),
                "stdk": (TopKSameClassMetric,{}), "groupk": (TopKSameSubclassMetric,{}),
                "switched": (SwitchMetric,{}),
                "add_batch_in": (BatchRetraining,{ **retrain_dict,**{"mode":"cum"}}),
                "add_batch_in_neg": (BatchRetraining, { **retrain_dict,**{"mode":"neg_cum"}}), 
                "leave_out": (BatchRetraining, { **retrain_dict,**{"mode":"leave_batch_out"}}),
                "only_batch": (BatchRetraining, { **retrain_dict,**{"mode":"single_batch"}}),
                "lds": (LinearDatamodelingScore, retrain_dict), "labelflip": (LabelFlipMetric, retrain_dict)}
    if metric_name not in ret_dict.keys():
        raise Exception(f"{metric_name} is not a metric name")
    metric_cls, metric_kwargs = ret_dict[metric_name]
    base_dict.update(metric_kwargs)
    return metric_cls(**base_dict)


def evaluate(model_name, model_path, device, class_groups,
             dataset_name, metric_name,
             data_root, xpl_root, coef_root,
             save_dir, validation_size, num_classes,
             epochs, loss, lr, momentum, optimizer, scheduler,
             weight_decay, augmentation):
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
    if metric_name in ["corrupt", "mark", "switched", "group"]:
        dataset_type=metric_name
    elif metric_name == "groupk":
        dataset_type="group"
    else:
        dataset_type = "std"
    train, test = load_datasets_reduced(dataset_name, dataset_type, ds_kwargs)
    model = load_model(model_name, dataset_name, num_classes).to(device)
    #if dataset_type == 'mark': # do we need these lines? not clear yet.
    #    checkpoint = torch.load(model_path, map_location=device)
    #    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    metric = load_metric(metric_name, dataset_name, train, test, device, coef_root, model, model_name,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation)
    print(f"Computing metric {metric.name}")

    if metric_name == 'switched':
        xpl_root_switched = xpl_root
        xpl_root = xpl_root.replace('switched', 'std')

        if not os.path.isdir(xpl_root):
            raise Exception(f"Can not find standard explanation directory {xpl_root}")
        file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f) and (".shark" not in f) and ("_all" not in f)]
        file_root = file_list[0].split('_')[0]
        num_files=len(file_list)
        #check if merged xpl exists
        if os.path.isfile(os.path.join(xpl_root, f"{file_root}_all")):
            xpl_all = torch.load(os.path.join(xpl_root, f"{file_root}_all"))
        #merge all xpl
        else:
            xpl_all = torch.empty(0, device=device)
            for i in range(num_files):
                fname = os.path.join(xpl_root, f"{file_root}_{i:02d}")
                xpl = torch.load(fname, map_location=torch.device(device))
                xpl.to(device)
                xpl_all = torch.cat((xpl_all, xpl), 0)
            torch.save(xpl_all, os.path.join(xpl_root, f"{file_root}_all"))

        if not os.path.isdir(xpl_root_switched):
            raise Exception(f"Can not find switched explanation directory {xpl_root_switched}")
        file_list_switched = [f for f in os.listdir(xpl_root_switched) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f) and (".shark" not in f) and ("_all" not in f)]
        file_root_switched = file_list_switched[0].split('_')[0]
        num_files_switched=len(file_list_switched)        
        #check if merged switched xpl exists
        if os.path.isfile(os.path.join(xpl_root_switched, f"{file_root_switched}_all")):
            xpl_all_switched = torch.load(os.path.join(xpl_root_switched, f"{file_root_switched}_all"))        
        #merge all switched xpl
        else:
            xpl_all_switched = torch.empty(0, device=device)
            for i in range(num_files_switched):
                fname_switched = os.path.join(xpl_root_switched, f"{file_root_switched}_{i:02d}")
                xpl_switched = torch.load(fname_switched, map_location=torch.device(device))
                xpl_switched.to(device)
                xpl_all_switched = torch.cat((xpl_all_switched, xpl_switched), 0)
            torch.save(xpl_all_switched, os.path.join(xpl_root_switched, f"{file_root_switched}_all"))

        metric(xpl_all, xpl_all_switched, 0)
        metric.get_result(save_dir, f"{dataset_name}_{metric_name}_{xpl_root.split('/')[-1]}_eval_results.json")

    else:
        if not os.path.isdir(xpl_root):
            raise Exception(f"Can not find standard explanation directory {xpl_root}")
        file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f) and (".shark" not in f) and ("_all" not in f)]
        file_root = file_list[0].split('_')[0]
        num_files=len(file_list)
        #check if merged xpl exists
        if metric_name == "corrupt":
            pass
        else:
            if os.path.isfile(os.path.join(xpl_root, f"{file_root}_all")):
                xpl_all = torch.load(os.path.join(xpl_root, f"{file_root}_all"))
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
            metric.get_result(save_dir, f"{dataset_name}_{metric_name}_{xpl_root.split('/')[-1]}_eval_results.json")


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

    evaluate(
                model_name=train_config.get('model_name', None),
                model_path=train_config.get('model_path', None),
                device=train_config.get('device', 'cuda'),
                class_groups=train_config.get('class_groups', None),
                dataset_name=train_config.get('dataset_name', None),
                metric_name=train_config.get('metric', 'std'),
                data_root=train_config.get('data_root', None),
                xpl_root=train_config.get('xpl_root', None),
                coef_root=train_config.get('coef_root', None),
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
                augmentation=train_config.get('augmentation', None)
    )