import argparse
from utils.data import load_datasets
from utils.models import load_model, load_cifar_model
import yaml
import logging
from metrics import *


def load_metric(dataset_type, dataset_name, train, test, device, coef_root, model,
                epochs, loss, lr, momentum, optimizer, scheduler,
                weight_decay, augmentation,):
    ret_dict = {"std": SameClassMetric, "group": SameSubclassMetric, "corrupt": CorruptLabelMetric,
                "mark": MarkImageMetric,
                "stdk": TopKSameClassMetric, "groupk": TopKSameSubclassMetric,
                "switched": SwitchMetric,
                "add_batch_in": CumAddBatchIn, "add_batch_in_neg": CumAddBatchInNeg, 
                "leave_out": LeaveBatchOut, "only_batch": OnlyBatch,
                "lds": LinearDatamodelingScore, "labelflip": LabelFlipMetric}
    if dataset_type not in ret_dict.keys():
        return SameClassMetric(train, test, device)
    metric_cls = ret_dict[dataset_type]
    if dataset_type == "corrupt":
        ret = metric_cls(train, test, coef_root, device)
    elif dataset_type == "mark":
        ret = metric_cls(train, test, model, device)
    elif dataset_type == "switched":
        ret = metric_cls(device)
    elif dataset_type in ["add_batch_in", "add_batch_in_neg", "leave_out", "only_batch", "lds", "labelflip"]:
        ret = metric_cls(dataset_name, train, test, model,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, device=device)
    else:
        ret = metric_cls(train, test, device=device)
    return ret


def evaluate(model_name, model_path, device, class_groups,
             dataset_name, dataset_type,
             data_root, xpl_root, coef_root,
             save_dir, validation_size, num_classes,
             epochs, loss, lr, momentum, optimizer, scheduler,
             weight_decay, augmentation,
             from_checkpoint=True):
    if not torch.cuda.is_available():
        device="cpu"
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "test",
        'validation_size': validation_size,
        'only_train': False,
        'transform': None
    }
    train, test = load_datasets(dataset_name, dataset_type, **ds_kwargs)
    if dataset_name == "CIFAR":
        model = load_cifar_model(model_path, dataset_type, num_classes, device)
    else:
        model = load_model(model_name, dataset_name, num_classes).to(device)
        if dataset_type == 'mark':
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    metric = load_metric(dataset_type, dataset_name, train, test, device, coef_root, model,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation)
    print(f"Computing metric {metric.name}")

    if dataset_type == 'switched':
        xpl_root_switched = xpl_root
        xpl_root = xpl_root.replace('switched', 'std')

        #merge all xpl
        if not os.path.isdir(xpl_root):
            raise Exception(f"Can not find standard explanation directory {xpl_root}")
        file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f) and (".shark" not in f) and ("_all" not in f)]
        file_root = file_list[0].split('_')[0]
        num_files=len(file_list)
        xpl_all = torch.empty(0, device=device)
        for i in range(num_files):
            fname = os.path.join(xpl_root, f"{file_root}_{i}")
            xpl = torch.load(fname, map_location=torch.device(device))
            xpl.to(device)
            xpl_all = torch.cat((xpl_all, xpl), 0)
        torch.save(xpl_all, os.path.join(xpl_root, f"{file_root}_all"))

        #merge all switched xpl
        if not os.path.isdir(xpl_root_switched):
            raise Exception(f"Can not find switched explanation directory {xpl_root_switched}")
        file_list_switched = [f for f in os.listdir(xpl_root_switched) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f) and (".shark" not in f) and ("_all" not in f)]
        file_root_switched = file_list_switched[0].split('_')[0]
        num_files_switched=len(file_list_switched)
        xpl_all_switched = torch.empty(0, device=device)
        for i in range(num_files_switched):
            fname_switched = os.path.join(xpl_root_switched, f"{file_root_switched}_{i}")
            xpl_switched = torch.load(fname_switched, map_location=torch.device(device))
            xpl_switched.to(device)
            xpl_all_switched = torch.cat((xpl_all_switched, xpl_switched), 0)
        torch.save(xpl_all_switched, os.path.join(xpl_root_switched, f"{file_root_switched}_all"))

        metric(xpl_all, xpl_all_switched, 0)
        metric.get_result(save_dir, f"{dataset_name}_{dataset_type}_{xpl_root.split('/')[-1]}_eval_results.json")

    else:
        #merge all xpl
        if not os.path.isdir(xpl_root):
            raise Exception(f"Can not find standard explanation directory {xpl_root}")
        file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f) and (".shark" not in f)]
        file_root = file_list[0].split('_')[0]
        num_files=len(file_list)
        xpl_all = torch.empty(0, device=device)
        for i in range(num_files):
            fname = os.path.join(xpl_root, f"{file_root}_{i}")
            xpl = torch.load(fname, map_location=torch.device(device))
            xpl.to(device)
            xpl_all = torch.cat((xpl_all, xpl), 0)
        torch.save(xpl_all, os.path.join(xpl_root, f"{file_root}_all"))
            
        metric(xpl_all, 0)
        metric.get_result(save_dir, f"{dataset_name}_{dataset_type}_{xpl_root.split('/')[-1]}_eval_results.json")


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

    evaluate(model_name=train_config.get('model_name', None),
                model_path=train_config.get('model_path', None),
                device=train_config.get('device', 'cuda'),
                class_groups=train_config.get('class_groups', None),
                dataset_name=train_config.get('dataset_name', None),
                dataset_type=train_config.get('metric', 'std'),
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