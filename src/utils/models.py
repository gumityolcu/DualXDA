import torch.utils.data
from models import BasicConvModel
from torchvision.models.resnet import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from transformers import AutoModelForSequenceClassification
from transformers.pytorch_utils import Conv1D
import tqdm
from torch import nn
import os
def clear_resnet_from_checkpoints(checkpoint):
    checkpoint["model_state"]={
            key:value for key, value in checkpoint["model_state"].items()
         if "resnet" not in key
        }
    return checkpoint

@torch.no_grad()
def replace_conv1d_modules(model: nn.Module) -> None:
    # GPT-2 is defined in terms of Conv1D. However, this does not work for Kronfluence.
    # Here, we convert these Conv1D modules to linear modules recursively.
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            new_module = nn.Linear(in_features=module.weight.shape[0], out_features=module.weight.shape[1])
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)

class GPT2Features(torch.nn.Module):
    def __init__(self, model, device):
        super(GPT2Features, self).__init__()
        self.device=device
        self.model = model.transformer
        self.model.to(device)
    
    def forward(self, batch):
        input_ids = batch[:, 0,:]
        attention_mask = batch[:, 1,:]
        transformer_outputs=self.model(input_ids=input_ids, attention_mask=attention_mask)
        features = transformer_outputs[0]
        last_non_pad_token = attention_mask.sum(dim=1) - 1
        features = features[torch.arange(features.shape[0], device=features.device), last_non_pad_token]
        return features

class GPT2Wrapper(torch.nn.Module):
    def __init__(self, hf_id, device):
        super(GPT2Wrapper, self).__init__()
        # model_ids={
        #     "tweet_sentiment_extraction": "herrerovir/gpt2-tweet-sentiment-model",
        #     "ag_news": "MoritzWeckbecker/gpt2-large_ag-news_full"
        # }
        self.device=device
        model = AutoModelForSequenceClassification.from_pretrained(hf_id)
        self.features=GPT2Features(model,device)
        replace_conv1d_modules(self.features)
        self.classifier=model.score
        # replace_conv1d_modules(self.classifier)
        self.classifier.to(device)
    
    def forward(self, batch):
        features=self.features(batch)
        return self.classifier(features)
    
    def influence_named_parameters(self):
       return [("classifier.weight", self.classifier.weight)]

    def select_arnoldi_params(self, name):
        if "features.model.h." in name:
            id=int(name.split(".")[3])
            return id>-1
        return False

    def arnoldi_parameters(self):
        return [n for n,_ in self.named_modules() if self.select_arnoldi_params(n)]


class LlamaFeatures(torch.nn.Module):
    def __init__(self, model, pad_token_id, device):
        super(LlamaFeatures, self).__init__()
        self.device=device
        self.model = model.model
        self.model.to(device)
        self.pad_token_id=pad_token_id
    
    def forward(self, batch):
        input_ids = batch[:, 0,:]
        attention_mask = batch[:, 1,:]
        transformer_outputs=self.model(input_ids=input_ids, attention_mask=attention_mask)
        features = transformer_outputs[0]
        non_pad_mask = (input_ids != self.pad_token_id).to(self.device, torch.int32)
        token_indices = torch.arange(input_ids.shape[-1], device=self.device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        features = features[torch.arange(features.shape[0], device=features.device), last_non_pad_token]
        return features

class LlamaWrapper(torch.nn.Module):
    def __init__(self, hf_id, device):
        super(LlamaWrapper, self).__init__()
        # model_ids={
        #     "tweet_sentiment_extraction": "herrerovir/gpt2-tweet-sentiment-model",
        #     "ag_news": "MoritzWeckbecker/gpt2-large_ag-news_full"
        # }
        self.device=device
        model = AutoModelForSequenceClassification.from_pretrained(hf_id)
        # model.config.pad_token_id=128001
        self.features=LlamaFeatures(model=model,device=device, pad_token_id=128001)
        # replace_conv1d_modules(self.features)
        self.classifier=model.score
        # replace_conv1d_modules(self.classifier)
        self.classifier.to(device)
        
    
    def forward(self, batch):
        features=self.features(batch)
        return self.classifier(features)
    
    def influence_named_parameters(self):
       return [("classifier.weight", self.classifier.weight)]

    # def select_arnoldi_params(self, name):
    #     if "features.model.h." in name:
    #         id=int(name.split(".")[3])
    #         return id>-1
    #     return False

    # def arnoldi_parameters(self):
    #     return [n for n,_ in self.named_modules() if self.select_arnoldi_params(n)]



class ResNetWrapper(torch.nn.Module):
    def __init__(self, module, output_dim, arnoldi_param_filter=None):
        # total=0
        # for p in module.parameters():
        #     num=1
        #     for s in p.data.shape:
        #         num=num*s
        #     total=total+num
        # print(total)
        # exit()
        super(ResNetWrapper, self).__init__()
        self.classifier=torch.nn.Linear(in_features=module.fc.in_features, out_features=output_dim, bias=True)
        self.arnoldi_param_filter=arnoldi_param_filter
        seq_array=[
            torch.nn.Conv2d(3, module.bn1.num_features,kernel_size=3,padding=2,stride=1),
            module.bn1,
            module.relu,
            module.maxpool,
            module.layer1,module.layer2,module.layer3,
            module.layer4,module.avgpool,
            torch.nn.Flatten()
            ]

        self.features=torch.nn.Sequential(*seq_array)

    def forward(self, x):
        x=self.features(x)
        return self.classifier(x)

    def influence_named_parameters(self):
       return [("classifier.weight", self.classifier.weight), ("classifier.bias", self.classifier.bias)]
    
    def arnoldi_parameters(self):
        if self.arnoldi_param_filter is None:           
            return None #None means we use all parameters. This is needed for full model explanation with CIFAR
        return ["classifier", "features.7.2.conv2"]
         

    def sim_parameters(self):
        return self.parameters()
    
def load_model(model_name, dataset_name, num_classes):
    if dataset_name=="MNIST":
        bias = not ('homo' in model_name)
        params={
                    'convs': {
                            'num': 3,
                            'padding': 0,
                            'kernel': 3,
                            'stride': 1,
                            'features': [5, 10, 5]
                        },

                    'fc' : {
                        'num': 2,
                        'features': [500, 100]
                    },

                    'input_shape':(1,28,28)
                }

        return BasicConvModel(input_shape=params['input_shape'], convs=params['convs'], fc=params['fc'], num_classes=num_classes)

    elif model_name=="resnet18":
        if dataset_name=="AWA":
            return ResNetWrapper(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), output_dim=num_classes, arnoldi_param_filter=True)
        else:
            return ResNetWrapper(resnet18(), output_dim=num_classes)
    
    else:
        if dataset_name=="AWA":
            return ResNetWrapper(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1), output_dim=num_classes, arnoldi_param_filter=True)
        else:
            return ResNetWrapper(resnet50(), output_dim=num_classes)


def compute_accuracy(model, test, device, progress=True):
    loader=torch.utils.data.DataLoader(test, 64, shuffle=False)
    acc = 0.
    model.eval()
    index = 0
    fault_list = []

    for x, y in tqdm.tqdm(loader, disable=not progress):
        x=x.to(device)
        if isinstance(y,list):
            y=y[1]
        y=y.to(device)
        real_out = torch.argmax(model(x), dim=1)
        preds = (y == real_out)
        racc = torch.sum(preds)
        if racc != 64:
            for j in range(x.size(0)):
                if not preds[j]:
                    fault_list.append(j + index)
        acc = acc + racc
        index += 64
    NNN = len(test)
    acc=acc/NNN
    return acc, fault_list
