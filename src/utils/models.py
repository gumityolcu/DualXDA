import torch.utils.data
import torchvision
from models import BasicConvModel, CIFARResNet, AWAResNet
from torchvision.models.resnet import resnet18, resnet50
import tqdm

class ResNetWrapper(torch.nn.Module):
    def __init__(self, module, output_dim):
        super().__init__()
        self.resnet=module
        self.classifier=torch.nn.Linear(in_features=module.fc.in_features, out_features=output_dim, bias=True)
        seq_array=[
            torch.nn.Conv2d(3,module.bn1.num_features,kernel_size=3,padding=2,stride=1),
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
    
def load_model(model_name, dataset_name, num_classes):
    if dataset_name=="MNIST":
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
                },
        return BasicConvModel(input_shape=params['input_shape'], convs=params['convs'], fc=params['fc'], num_classes=num_classes, bias=True)
    if model_name=="resnet18":
        return ResNetWrapper(resnet18(),output_dim=num_classes)
    else:
        return ResNetWrapper(resnet50(),output_dim=num_classes)

def load_cifar_model(model_path,dataset_type,num_classes,device, train=False):
    model=resnet18()
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=False)
    if train==False:
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint={key[6:]: value for key,value in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint)
    model.eval()
    return CIFARResNet(model,device=device)

def load_awa_model(model_path,dataset_type,num_classes,device,train=False):
    model=resnet50(model_path)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=False)
    if train == False:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    model.eval()
    return AWAResNet(model,device=device)


def compute_accuracy(model, test, device):
    loader=torch.utils.data.DataLoader(test, 64, shuffle=False)
    acc = 0.
    model.eval()
    index = 0
    fault_list = []

    for x, y in tqdm.tqdm(loader):
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
