import torch.utils.data
from models import BasicConvModel
from torchvision.models.resnet import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models import vgg16, VGG16_Weights
from torch.nn import Flatten
import tqdm

def clear_resnet_from_checkpoints(checkpoint):
    checkpoint["model_state"]={
            key:value for key, value in checkpoint["model_state"].items()
         if "resnet" not in key
        }
    return checkpoint

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
    
class VGGWrapper(torch.nn.Module):
    def __init__(self, module, output_dim, arnoldi_param_filter=None):
        super(VGGWrapper, self).__init__()
        self.classifier=torch.nn.Linear(in_features=module.classifier[6].in_features, out_features=output_dim, bias=True)
        self.arnoldi_param_filter=arnoldi_param_filter
        seq_array= [module.features[i] for i in range(len(module.features))] + [
            module.avgpool,
            Flatten(1),
            module.classifier[0],
            module.classifier[1],
            module.classifier[2],
            module.classifier[3],
            module.classifier[4],
            module.classifier[5],
            ]

        self.features=torch.nn.Sequential(*seq_array)

    def forward(self, x):
        x=self.features(x)
        return self.classifier(x)

    def influence_named_parameters(self):
       return [("classifier.weight", self.classifier.weight), ("classifier.bias", self.classifier.bias)]        

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
    
    elif model_name=="resnet50":
        if dataset_name=="AWA":
            return ResNetWrapper(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1), output_dim=num_classes, arnoldi_param_filter=True)
        else:
            return ResNetWrapper(resnet50(), output_dim=num_classes)
    elif model_name=="vgg16":
        print("Using VGG16 model\n")
        if dataset_name=="AWA":
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            # Freeze parameters of early layers (up to and incl. first linear layer)
            for layer in vgg.features:
                for param in layer.parameters():
                    param.requires_grad = False
            for params in vgg.avgpool.parameters():
                params.requires_grad = False
            for params in vgg.classifier[0].parameters():
                params.requires_grad = False
            return VGGWrapper(vgg, output_dim=num_classes, arnoldi_param_filter=True)
        else:
            vgg = vgg16()
            # Freeze parameters of early layers (up to and incl. first linear layer)
            for layer in vgg.features:
                for param in layer.parameters():
                    param.requires_grad = False
            for params in vgg.avgpool.parameters():
                params.requires_grad = False
            for params in vgg.classifier[0].parameters():
                params.requires_grad = False
            return VGGWrapper(vgg, output_dim=num_classes)        


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