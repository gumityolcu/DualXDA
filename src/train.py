import argparse
import math
import logging
from models import BasicConvModel
import os
import json
import sys
import yaml
from torch.nn import CrossEntropyLoss, KLDivLoss, BCEWithLogitsLoss, MultiMarginLoss, BCELoss
from torch.nn.functional import one_hot
from tqdm import tqdm
from utils.models import load_model
from utils.data import ReduceLabelDataset, FeatureDataset, GroupLabelDataset, CorruptLabelDataset
import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomApply, RandomEqualize, RandomRotation, AutoAugment, AutoAugmentPolicy
import matplotlib.pyplot as plt
from dataset.MNIST import MNIST
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ConstantLR, StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from utils.data import load_datasets_reduced, RestrictedDataset
import warnings


def test_models_in_dir(model_root_path, model_name, device, num_classes, class_groups, data_root, batch_size,
                   num_batches_to_process, dataset_name, dataset_type, validation_size, save_dir):
    #model_root_path=os.path.join(model_root_path, dataset_name)
    filelist=[file for file in os.listdir(model_root_path) if not os.path.isdir(os.path.join(model_root_path, file)) and not ".out" in file and not ".tgz" in file]
    
    # model1=load_model("basic_conv", "MNIST", 10)
    # model2=load_model("basic_conv", "MNIST", 10)
    # model1.load_state_dict(torch.load(os.path.join(model_root_path,"MNIST_basic_conv_119"),map_location="cpu")["model_state"])
    # model2.load_state_dict(torch.load(os.path.join(model_root_path,"MNIST_basic_conv_149"),map_location="cpu")["model_state"])
    # for name, param1 in model1.named_parameters():
    #     param2=model1.state_dict()[name]
    #     pass
    results_dict={}
    for fname in filelist:
        print(f"Testing: {fname}")       
        results_dict[fname]={}
        spl=fname.split("_")
        model_path=os.path.join(model_root_path, fname)
        dataset_type=fname.split("_")[-1]
        num_classes=5 if dataset_type=="group" else 10 
        train_acc=evaluate_model(
            model=model_name,
            device=device,
            num_classes=num_classes,
            class_groups=class_groups,
            data_root=data_root,
            batch_size=batch_size,
            num_batches_to_process=num_batches_to_process,
            load_path=model_path,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            validation_size=validation_size,
            image_set="train"
        )
        test_acc = evaluate_model(
            model=model_name,
            device=device,
            num_classes=num_classes,
            class_groups=class_groups,
            data_root=data_root,
            batch_size=batch_size,
            num_batches_to_process=num_batches_to_process,
            load_path=model_path,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            validation_size=validation_size,
            image_set="test"
        )
        print(f"train: {train_acc} - test: {test_acc}")
        results_dict[fname]["train"]=train_acc.item()
        results_dict[fname]["test"]=test_acc.item()
    save_dir=os.path.join(save_dir,"results.json")
    with open(save_dir, 'w') as file:
        json.dump(results_dict, file)
    return results_dict

def parse_report(rep, num_classes):
    print(rep)
    ret = dict()
    rep = rep.split('\n')
    keys = rep[0].strip().split(' ')
    i = 0
    while i < len(keys):
        k = keys[i]
        if k == '':
            if i != len(keys) - 1:
                keys = keys[:i] + keys[i + 1:]
                i = i - 1
            else:
                keys = keys[:-1]
        i = i + 1
    keys = keys[:-1]
    for k in keys:
        ret[k] = dict()
    rep = rep[num_classes + 3:-1]
    for line in rep:
        for i, key in enumerate(keys):
            spl = line.strip().split('    ')
            ret[key][spl[0].strip().replace(' ', '_')] = float(spl[i + 1].strip())
    return ret

def get_validation_loss(model, ds, loss, num_classes, device):
    model.eval()
    #loader = DataLoader(ds, batch_size=64)
    loader = DataLoader(ds, batch_size=32)
    l = torch.tensor(0.0)
    # count = 0
    #for inputs, targets in tqdm(iter(loader)): #Disable for cluster
    for inputs, targets in iter(loader):
        inputs = inputs.to(torch.device(device))
        targets = targets.long()
        if isinstance(loss, BCEWithLogitsLoss):
            targets = one_hot(targets, num_classes).float()
        targets = targets.to(torch.device(device))
        with torch.no_grad():
            y = model(inputs)
            l = l + loss(y, targets)
        # count = count + inputs.shape[0]
    # l = l / count
    model.train()
    return l

def load_scheduler(name, optimizer, epochs): #include warmup?
    if name == "constant":
        scheduler = ConstantLR(optimizer=optimizer, factor=1.0, last_epoch=-1)
    elif name == "annealing":
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max = epochs, last_epoch=epochs)
    elif name == "step":
        scheduler = StepLR(optimizer=optimizer, step_size=epochs // 20, gamma=0.1, last_epoch=epochs)
    else:
        scheduler = ConstantLR(optimizer=optimizer, factor=1.0, last_epoch=-1)
    return scheduler        

def load_optimizer(name, model, lr, weight_decay, momentum): #could add momentum as a variable
    optimizer_dict = {
        "sgd": SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum),
        "adam": Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas = (momentum, 0.999)), #No momentum for ADAM
        "rmsprop": RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    }
    optimizer = optimizer_dict.get(name, SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum))
    if name == "adam":
        warnings.warn("For Adam, the given momentum value is used for beta_1.")
    return optimizer

def load_loss(name): #add regularisation
    loss_dict = {
        "cross_entropy": CrossEntropyLoss(),
        "bce": BCEWithLogitsLoss(reduction='sum'),
        "hinge": MultiMarginLoss()
    }
    loss = loss_dict.get(name, CrossEntropyLoss())
    return loss

def load_augmentation(name, dataset_name):
    if name is None:
        return lambda x:x
    shapes={
        "MNIST": (28,28),
        "CIFAR": (32,32),
        "AWA": (224,224),
        "Shapes2x2": (224, 224)
    }
    trans_arr=[]
    trans_dict={
        "crop": RandomApply([RandomResizedCrop(size=shapes[dataset_name], )], p=0.5),
        "flip": RandomHorizontalFlip(),
        "eq": RandomEqualize(),
        "rotate": RandomApply([RandomRotation(degrees=(0,180))],p=0.5),
        "cifar": AutoAugment(AutoAugmentPolicy.CIFAR10),
        "imagenet": AutoAugment(AutoAugmentPolicy.IMAGENET)
    }
    for trans in name.split("_"):
        if trans in trans_dict.keys():
            trans_arr.append(trans_dict[trans])
    return Compose(trans_arr)

def start_training(model_name, device, num_classes, class_groups, data_root, epochs,
                   batch_size, lr, weight_decay, momentum, save_dir, save_each, model_path, base_epoch,
                   dataset_name, dataset_type, num_batches_eval, validation_size,
                   augmentation, optimizer, scheduler, loss, train_indices=None):
    if not torch.cuda.is_available():
        device="cpu"
    if dataset_type=="group":
        num_classes_model=len(class_groups)
    else: 
        num_classes_model = num_classes
    model = load_model(model_name, dataset_name, num_classes_model).to(device)
    tensorboarddir = f"{model_name}_{lr}_{scheduler}_{optimizer}{f'_aug' if augmentation is not None else ''}"
    tensorboarddir = os.path.join(save_dir, tensorboarddir)
    writer = SummaryWriter(tensorboarddir)

    learning_rates=[]
    train_losses = []
    validation_losses = []
    validation_epochs = []
    val_acc = []
    train_acc = []
    loss=load_loss(loss)
    optimizer = load_optimizer(optimizer, model, lr, weight_decay, momentum)

    initial_lr = optimizer.param_groups[0]['lr']
    print(f"Before scheduler: {initial_lr}")
    scheduler = load_scheduler(scheduler, optimizer, epochs)
    after_scheduler_lr = optimizer.param_groups[0]['lr']
    print(f"After scheduler creation: {after_scheduler_lr}")

    if augmentation not in [None, '']:
        augmentation = load_augmentation(augmentation, dataset_name)

    kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "val",
        'validation_size': validation_size,
        'only_train': True,
        'transform': augmentation,
        'num_classes': num_classes
    }
    corrupt = (dataset_type == "corrupt")
    group = (dataset_type == "group")
    ds, valds = load_datasets_reduced(dataset_name, dataset_type, kwargs)
    if train_indices is not None:
        ds = RestrictedDataset(ds, train_indices, return_indices=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    saved_files = []

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        if checkpoint.get("model_state", None) != None:
            model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state", None) != None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state", None) != None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("train_losses", None) != None:
            train_losses = checkpoint.get("train_losses", None)
        if checkpoint.get("validation_losses", None) != None:    
            validation_losses = checkpoint.get("validation_losses", None)
        if checkpoint.get("validation_epochs", None) != None:
            validation_epochs = checkpoint.get("validation_epochs", None)
        if checkpoint.get("validation_accuracy", None) != None:
            val_acc = checkpoint.get("validation_accuracy", None)
        if checkpoint.get("train_accuracy", None) != None:
            train_acc = checkpoint.get("train_accuracy", None)

    for i,r in enumerate(learning_rates):
        writer.add_scalar('Metric/lr', r, i)
    for i, r in enumerate(train_acc):
        writer.add_scalar('Metric/train_acc', r, i)
    for i, r in enumerate(val_acc):
        writer.add_scalar('Metric/val_acc', r, validation_epochs[i])
    for i, l in enumerate(train_losses):
        writer.add_scalar('Loss/train', l, i)
    for i, l in enumerate(validation_losses):
        writer.add_scalar('Loss/val', l, validation_epochs[i])

    model.train()
    best_model_yet = model_path
    best_loss_yet = None

    print("Device:", device)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    for e in range(epochs):
        print(f"Epoch {e + 1}/{epochs}", flush=True)
        y_true = torch.empty(0, device=device)
        y_out = torch.empty((0, num_classes_model), device=device)
        cum_loss = 0
        cnt = 0
        #for inputs, targets in tqdm(iter(loader)): # Disable for cluster
        for inputs, targets in iter(loader):
            inputs = inputs.to(device)
            if isinstance(loss, BCEWithLogitsLoss):
                targets = one_hot(targets, num_classes_model).float()
            targets = targets.to(device)

        
            y_true = torch.cat((y_true, targets), 0)

            optimizer.zero_grad()
            logits = model(inputs)
            l = loss(logits, targets)
            y_out = torch.cat((y_out, logits.detach().clone()), 0)
            if math.isnan(l):
                if not os.path.isdir("./broken_model"):
                    os.mkdir("broken_model")
                save_dict = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch': base_epoch + e,
                    'learning_rates': learning_rates,
                    'train_losses': train_losses,
                    'validation_losses': validation_losses,
                    'validation_epochs': validation_epochs,
                    'validation_accuracy': val_acc,
                    'ds_type':dataset_type,
                    'train_accuracy': train_acc
                }
                path = os.path.join("./broken_model", f"{dataset_name}_{model_name}_{base_epoch + e}")
                torch.save(save_dict, path)
                print("NaN loss")
                exit()
            l.backward()
            optimizer.step()
            cum_loss = cum_loss + l
            cnt = cnt + inputs.shape[0]
        #y_out = torch.softmax(y_out, dim=1)
        y_pred = torch.argmax(y_out, dim=1)
        # y_true = y_true.cpu().numpy()
        # y_out = y_out.cpu().numpy()
        # y_pred = y_pred.cpu().numpy()
        train_loss = cum_loss.detach().cpu()
        acc = (y_true == y_pred).sum() / y_out.shape[0]
        train_acc.append(acc)
        print(f"train accuracy: {acc}")
        writer.add_scalar('Metric/train_acc', acc, base_epoch + e)
        #writer.add_scalar('Metric/learning_rates', 0.95, base_epoch + e)
        train_losses.append(train_loss)
        writer.add_scalar('Loss/train', train_loss, base_epoch + e)
        print(f"Epoch {e + 1}/{epochs} loss: {cum_loss}")  # / cnt}")
        print("\n==============\n")
        learning_rates.append(scheduler.get_lr())
        current_lr = scheduler.get_lr() if hasattr(scheduler, 'get_lr') else [group['lr'] for group in optimizer.param_groups]
        print(f"Current learning rate: {current_lr}")
        scheduler.step()
        if (e + 1) % save_each == 0:
            validation_loss = get_validation_loss(model, valds, loss, num_classes_model, device)
            validation_losses.append(validation_loss.detach().cpu())
            validation_epochs.append(e)
            save_dict = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': base_epoch + e,
                'train_losses': train_losses,
                'validation_losses': validation_losses,
                'validation_epochs': validation_epochs,
                'validation_accuracy': val_acc,
                'learning_rates': learning_rates,
                'train_accuracy': train_acc
            }
            if corrupt:
                save_dict["corrupt_samples"] = ds.dataset.corrupt_samples
                save_dict["corrupt_labels"] = ds.dataset.corrupt_labels
            if group:
                save_dict["classes"] = ds.dataset.classes
            save_id = f"{dataset_name}_{model_name}_{base_epoch + e}"
            model_save_path = os.path.join(save_dir, save_id)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torch.save(save_dict, model_save_path)
            saved_files.append((model_save_path, save_id))

            print(f"\n\nValidation loss: {validation_loss}\n\n")
            writer.add_scalar('Loss/val', validation_loss, base_epoch + e)
            valeval = evaluate_model(model=model, device=device, num_classes=num_classes,
                                     data_root=data_root,
                                     batch_size=batch_size, num_batches_to_process=num_batches_eval,
                                     load_path=model_save_path, dataset_name=dataset_name, dataset_type=dataset_type,
                                     validation_size=validation_size,
                                     image_set="val", class_groups=class_groups
                                     )
            print(f"validation accuracy: {valeval}")
            writer.add_scalar('Metric/val_acc', valeval, base_epoch + e)
            val_acc.append(valeval)
            if best_loss_yet is None or best_loss_yet > validation_loss:
                best_loss_yet = validation_loss
                path = os.path.join(save_dir, f"best_val_score_{dataset_name}_{model_name}_{base_epoch + e}")
                torch.save(save_dict, path)
                if best_model_yet is not None:
                    os.remove(best_model_yet)
                best_model_yet = path
        writer.flush()

        # Save train and validation loss figures
        # plt.subplot(2, 1, 1)
        # plt.title("Training Loss")
        # plt.plot(base_epoch + np.asarray(range(epochs)), np.asarray(train_losses))
        # plt.subplot(2, 1, 2)
        # plt.title("Validation Loss")
        # plt.plot(base_epoch + np.asarray(vaidation_epochs), np.asarray(validation_losses))
        # plt.savefig(os.path.join(save_dir, f"{dataset_name}_{model_name}_{base_epoch + epochs}_losses.png"))
    writer.close()
    save_id = os.path.basename(best_model_yet)

def evaluate_model(model, device, num_classes, class_groups, data_root, batch_size,
                   num_batches_to_process, load_path, dataset_name, dataset_type, validation_size, image_set):
    if not torch.cuda.is_available():
        device="cpu"
    if dataset_type=="group":
        num_classes_model=len(class_groups)
    else: 
        num_classes_model = num_classes
    if isinstance(model,str):
        model = load_model(model, dataset_name, num_classes_model).to(device)

    kwparams = {
        'data_root': data_root,
        'image_set': image_set,
        'class_groups': class_groups,
        'validation_size': validation_size,
        'only_train': True,
        'transform': None,
        'num_classes': num_classes
    }
    _, ds = load_datasets_reduced(dataset_name=dataset_name, dataset_type=dataset_type, kwparams=kwparams)
    if not len(ds) > 0:
        return 0.0, 0.0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if isinstance(model, str):
        if load_path is not None:
            checkpoint = torch.load(load_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
    # classes = ds.all_classes
    model.eval()
    y_true = torch.empty(0, device=device)
    y_out = torch.empty((0, num_classes_model), device=device)

    #for i, (inputs, targets) in enumerate(tqdm(iter(loader), total=min(num_batches_to_process, len(loader)))): #Disable for cluster
    for i, (inputs, targets) in enumerate(iter(loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        y_true = torch.cat((y_true, targets), 0)
        with torch.no_grad():
            logits = model(inputs)
        y_out = torch.cat((y_out, logits), 0)

    results = dict()
    results["tpr"] = dict()
    results["fpr"] = dict()
    results["roc_auc"] = dict()

    y_out = torch.softmax(y_out, dim=1)
    y_pred = torch.argmax(y_out, dim=1)
    # y_true = y_true.cpu().numpy()
    # y_out = y_out.cpu().numpy()
    # y_pred = y_pred.cpu().numpy()
    model.train()
    return (y_true == y_pred).sum() / y_out.shape[0]

def test_all_models(model_root_path, model_name, device, num_classes, class_groups, data_root, batch_size,
                   num_batches_to_process, dataset_name, dataset_type, validation_size, save_dir):
    model_root_path=os.path.join(model_root_path, dataset_name)
    dirlist=[direc for direc in os.listdir(model_root_path) if ((os.path.isdir(os.path.join(model_root_path, direc))) and (direc != "results_all"))]
    results_dict={}
    for dir_name in dirlist:
        print(f"Testing: {dir_name}")       
        results_dict[dir_name]={}
        spl=dir_name.split("_")
        model_path=os.path.join(model_root_path, dir_name)
        model_path=os.path.join(model_path,f"{dataset_name}_{model_name}")
        dataset_type=dir_name
        num_classes=5 if dataset_type=="group" else 10
        train_acc=evaluate_model(
            model=model_name,
            device=device,
            num_classes=num_classes,
            class_groups=class_groups,
            data_root=data_root,
            batch_size=batch_size,
            num_batches_to_process=num_batches_to_process,
            load_path=model_path,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            validation_size=validation_size,
            image_set="train"
        )
        test_acc = evaluate_model(
            model=model_name,
            device=device,
            num_classes=num_classes,
            class_groups=class_groups,
            data_root=data_root,
            batch_size=batch_size,
            num_batches_to_process=num_batches_to_process,
            load_path=model_path,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            validation_size=validation_size,
            image_set="test"
        )
        print(f"train: {train_acc} - test: {test_acc}")
        results_dict[dir_name]["train"]=train_acc
        results_dict[dir_name]["test"]=test_acc
    save_dir=os.path.join(save_dir,"results.json")
    with open(save_dir, 'w') as file:
        json.dump(results_dict, file)
    return results_dict



if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current)
    sys.path.append(current)
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

    # test_all_models(
    #    model_root_path="/home/fe/yolcu/Documents/Code/DualView-wip/checkpoints",
    #    model_name=train_config.get('model_name', 'basic_conv'),
    #    device=train_config.get('device', 'cuda'),
    #    num_classes=train_config.get('num_classes', None),
    #    class_groups=train_config.get('class_groups', None),
    #    dataset_name=train_config.get('dataset_name', None),
    #    dataset_type=train_config.get('dataset_type', 'std'),
    #    data_root=train_config.get('data_root', None),
    #    batch_size=train_config.get('batch_size', None),
    #    save_dir=train_config.get('save_dir', None),
    #    num_batches_to_process=train_config.get('num_batches_eval', None),
    #    validation_size=train_config.get('validation_size', 2000)
    # )
    # test_models_in_dir(
    #    model_root_path="/home/fe/yolcu/Documents/Code/DualView-wip/test_output/MNIST",
    #    model_name=train_config.get('model_name', 'basic_conv'),
    #    device=train_config.get('device', 'cuda'),
    #    num_classes=train_config.get('num_classes', None),
    #    class_groups=train_config.get('class_groups', None),
    #    dataset_name=train_config.get('dataset_name', None),
    #    dataset_type=train_config.get('dataset_type', 'std'),
    #    data_root=train_config.get('data_root', None),
    #    batch_size=train_config.get('batch_size', None),
    #    save_dir=train_config.get('save_dir', None),
    #    num_batches_to_process=train_config.get('num_batches_eval', None),
    #    validation_size=train_config.get('validation_size', 2000)
    # )
    # exit()

    start_training(model_name=train_config.get('model_name', None),
                   model_path=train_config.get('model_path', None),
                   base_epoch=train_config.get('base_epoch', 0),
                   device=train_config.get('device', 'cuda'),
                   num_classes=train_config.get('num_classes', None),
                   class_groups=train_config.get('class_groups', None),
                   dataset_name=train_config.get('dataset_name', None),
                   dataset_type=train_config.get('dataset_type', 'std'),
                   data_root=train_config.get('data_root', None),
                   epochs=train_config.get('epochs', None),
                   batch_size=train_config.get('batch_size', None),
                   lr=train_config.get('lr', 0.1),
                   momentum=train_config.get('momentum', 0.9),
                   weight_decay=train_config.get('weight_decay', 0),
                   augmentation=train_config.get('augmentation', None),
                   loss=train_config.get('loss', None),
                   optimizer=train_config.get('optimizer', None),
                   save_dir=train_config.get('save_dir', None),
                   save_each=train_config.get('save_each', 100),
                   num_batches_eval=train_config.get('num_batches_eval', None),
                   validation_size=train_config.get('validation_size', 2000),
                   scheduler=train_config.get('scheduler', None)
                   )
