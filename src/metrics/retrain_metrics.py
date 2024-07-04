from utils import Metric
import torch
import copy

class RetrainMetric(Metric):
    name = "RetrainMetric"
    
    def retrain(self, model, train_new):
        # copy model with blank weights
        retrained_model = copy.deepcopy(model)
        for layer in retrained_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # define retraining regiment
        retraining_loader = torch.utils.data.DataLoader(train_new, batch_size=4, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # retrain
        retrained_model.train()

        return retrained_model
    
# Parent Retrain Metric should only have method 'retrain' as scores etc are all different depending on the metric
    
class LeaveBatchOut(RetrainMetric):
    name = "LeaveBatchOutMetric"

    def __init__(self, train, test, model, batch, device="cuda"):
        self.train = train
        self.test = test
        self.model = model
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.batch = batch
        self.device = device

    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        
