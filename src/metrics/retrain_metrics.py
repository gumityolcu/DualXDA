from utils import Metric
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy
from utils.data import RestrictedDataset, FlipLabelDataset
from train import load_loss, load_optimizer, load_scheduler, load_augmentation
from tqdm import tqdm
import numpy as np
from torchmetrics.regression import SpearmanCorrCoef

class RetrainMetric(Metric):
    name = "RetrainMetric"
    
    def __init__(self, dataset_name, train, test, model,
                 epochs, loss, lr, momentum, optimizer, scheduler,
             weight_decay, augmentation, device):
        self.dataset_name = dataset_name
        self.train = train
        self.test = test
        self.num_classes = 10
        self.model = model #load model WITHOUT checkpoint in evaluate script for this metric!
        self.device = device
        self.batch_size = 64
        self.epochs = epochs
        self.lr = lr
        self.augmentation = augmentation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.weight_decay = weight_decay
        self.momentum = momentum
        if dataset_name == "AWA":
            self.num_classes = 50
        else:
            self.num_classes = 10
            
    def retrain(self, ds):
        learning_rates=[]
        train_losses = []

        model = copy.deepcopy(self.model).to(self.device)
        loss = load_loss(self.loss)
        optimizer = load_optimizer(self.optimizer, model, self.lr, self.weight_decay, self.momentum)
        scheduler = load_scheduler(self.scheduler, optimizer)
        augmentation = load_augmentation(self.augmentation, self.dataset_name)
        train_acc = []
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        model.train()

        for e in range(self.epochs):
            y_true = torch.empty(0, device=self.device)
            y_out = torch.empty((0, self.num_classes), device=self.device)
            cum_loss = 0
            cnt = 0
            for inputs, targets in tqdm(iter(loader)):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
                if self.augmentation is not None:
                    inputs=augmentation(inputs)

                y_true = torch.cat((y_true, targets), 0)

                optimizer.zero_grad()
                logits = model(inputs)
                l = loss(logits, targets)
                y_out = torch.cat((y_out, logits.detach().clone()), 0)
                l.backward()
                optimizer.step()
                cum_loss = cum_loss + l
                cnt = cnt + inputs.shape[0]
            y_pred = torch.argmax(y_out, dim=1)
            train_loss = cum_loss.detach().cpu()
            acc = (y_true == y_pred).sum() / y_out.shape[0]
            train_acc.append(acc)
            print(f"train accuracy: {acc}")
            #writer.add_scalar('Metric/train_acc', acc, base_epoch + e)
            #writer.add_scalar('Metric/learning_rates', 0.95, base_epoch + e)
            train_losses.append(train_loss)
            #writer.add_scalar('Loss/train', train_loss, base_epoch + e)
            print(f"Epoch {e + 1}/{self.epochs} loss: {cum_loss}")  # / cnt}")
            print("\n==============\n")
            learning_rates.append(scheduler.get_lr())
            scheduler.step()
        return model
    
    def evaluate(self, retrained_model, evalds, num_classes, batch_size=20):
        retrained_model.eval()
        loader = DataLoader(evalds, batch_size=batch_size, shuffle=True)
        y_true = torch.empty(0, device=self.device)
        y_out = torch.empty((0, self.num_classes), device=self.device)

        for i, (inputs, targets) in enumerate(tqdm(iter(loader), total=len(loader))):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            y_true = torch.cat((y_true, targets), 0)
            with torch.no_grad():
                logits = retrained_model(inputs)
            y_out = torch.cat((y_out, logits), 0)

        y_out = torch.softmax(y_out, dim=1)
        y_pred = torch.argmax(y_out, dim=1)
        return (y_true == y_pred).sum() / y_out.shape[0]
    

    
class CumAddBatchIn(RetrainMetric):
    name = "CumAddBatchIn"

    def __init__(self, dataset_name, train, test, model, 
                 epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, batch_nr=10, device="cuda"):
        super().__init__(dataset_name, train, test, model,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, device)
        self.batch_nr = batch_nr
        self.batchsize = len(self.train) // batch_nr
        self.loss_array = None

    def __call__(self, xpl, start_index=0, n_test=10):
        xpl.to(self.device)
        evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + n_test)], dim=0).to(self.device)
        evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + n_test)]).long().to(self.device)
        self.loss_array = np.empty((n_test, self.batch_nr))
        loss = CrossEntropyLoss()
        for test_index in range(n_test):
            for i in range(self.batch_nr):
                indices_sorted = xpl[test_index].argsort(descending=True)
                ds = RestrictedDataset(self.train, indices_sorted[:(i+1)*self.batchsize])
                retrained_model = self.retrain(ds)
                self.loss_array[test_index, i] = loss(retrained_model(evalds[start_index + test_index].unsqueeze(0)), evalds_labels[start_index + test_index].unsqueeze(0)).cpu().detach().numpy()

    def get_result(self, dir=None, file_name=None):
        # USE THIS WHEN MULTIPLE FILES FOR DIFFERENT XPL ARE READ IN
        #avg_scores = self.scores.mean(dim=0).to('cpu').detach().numpy()
        #self.scores = self.scores.to('cpu').detach().numpy()
        #resdict = {'metric': self.name, 'all_batch_scores': self.scores, 'all_batch_scores_avg': avg_scores,
        #           'scores_for_most_relevant_batch': self.scores[0], 'score_for_most_relevant_batch_avg': avg_scores[0],
        #           'num_batches': self.scores.shape[0]}
        self.scores = self.scores.to('cpu').detach().numpy()
        resdict = {'metric': self.name, 'all_batch_scores': self.loss_array,
                   'num_batches': self.scores.shape}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict

class CumAddBatchInNeg(RetrainMetric):
    # Add batches in from highest negative relevance to highest positive relevance
    name = "CumAddBatchInNeg"

    def __init__(self, dataset_name, train, test, model, epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, batch_nr=10, device="cuda"):
        super().__init__(dataset_name, train, test, model,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, device)
        self.batch_nr = batch_nr
        self.batchsize = len(self.train) // batch_nr
        self.loss_array = None

    def __call__(self, xpl, start_index=0, n_test=10):
        xpl.to(self.device)
        evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + n_test)], dim=0).to(self.device)
        evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + n_test)]).long().to(self.device)
        self.loss_array = np.empty((n_test, self.batch_nr))
        loss = CrossEntropyLoss()
        for test_index in range(n_test):
            for i in range(self.batch_nr):
                indices_sorted = xpl[test_index].argsort(descending=False)
                ds = RestrictedDataset(self.train, indices_sorted[:(i+1)*self.batchsize])
                retrained_model = self.retrain(ds)
                self.loss_array[test_index, i] = loss(retrained_model(evalds[start_index + test_index].unsqueeze(0)), evalds_labels[start_index + test_index].unsqueeze(0)).cpu().detach().numpy()
        

    def get_result(self, dir=None, file_name=None):
        # USE THIS WHEN MULTIPLE FILES FOR DIFFERENT XPL ARE READ IN
        #avg_scores = self.scores.mean(dim=0).to('cpu').detach().numpy()
        #self.scores = self.scores.to('cpu').detach().numpy()
        #resdict = {'metric': self.name, 'all_batch_scores': self.scores, 'all_batch_scores_avg': avg_scores,
        #           'scores_for_most_relevant_batch': self.scores[0], 'score_for_most_relevant_batch_avg': avg_scores[0],
        #           'num_batches': self.scores.shape[0]}
        self.scores = self.scores.to('cpu').detach().numpy()
        resdict = {'metric': self.name, 'all_batch_scores': self.loss_array,
                   'num_batches': self.scores.shape}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict

class LeaveBatchOut(RetrainMetric):
    name = "LeaveBatchOut"

    def __init__(self, dataset_name, train, test, model, epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, batch_nr=10, device="cuda"):
        super().__init__(dataset_name, train, test, model,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, device)
        self.batch_nr = batch_nr
        self.batchsize = len(self.train) // batch_nr
        self.loss_array = None

    def __call__(self, xpl, start_index=0, n_test=10):
        xpl.to(self.device)
        evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + n_test)], dim=0).to(self.device)
        evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + n_test)]).long().to(self.device)
        self.loss_array = np.empty((n_test, self.batch_nr))
        loss = CrossEntropyLoss()
        for test_index in range(n_test):
            for i in range(self.batch_nr):
                indices_sorted = xpl[test_index].argsort(descending=True)
                ds = RestrictedDataset(self.train, torch.cat((indices_sorted[:i*self.batchsize], indices_sorted[(i+1)*self.batchsize:])))
                retrained_model = self.retrain(ds)
                self.loss_array[test_index, i] = loss(retrained_model(evalds[start_index + test_index].unsqueeze(0)), evalds_labels[start_index + test_index].unsqueeze(0)).cpu().detach().numpy()
       

    def get_result(self, dir=None, file_name=None):
        # USE THIS WHEN MULTIPLE FILES FOR DIFFERENT XPL ARE READ IN
        #avg_scores = self.scores.mean(dim=0).to('cpu').detach().numpy()
        #self.scores = self.scores.to('cpu').detach().numpy()
        #resdict = {'metric': self.name, 'all_batch_scores': self.scores, 'all_batch_scores_avg': avg_scores,
        #           'scores_for_most_relevant_batch': self.scores[0], 'score_for_most_relevant_batch_avg': avg_scores[0],
        #           'num_batches': self.scores.shape[0]}
        self.scores = self.scores.to('cpu').detach().numpy()
        resdict = {'metric': self.name, 'all_batch_scores': self.loss_array,
                   'num_batches': self.scores.shape}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict
    
class OnlyBatch(RetrainMetric):
    name = "OnlyBatch"

    def __init__(self, dataset_name, train, test, model, epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, batch_nr=10, device="cuda"):
        super().__init__(dataset_name, train, test, model,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, device)
        self.batch_nr = batch_nr
        self.batchsize = len(self.train) // batch_nr
        self.loss_array = None

    def __call__(self, xpl, start_index=0, n_test=10):
        xpl.to(self.device)
        evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + n_test)], dim=0).to(self.device)
        evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + n_test)]).long().to(self.device)
        self.loss_array = np.empty((n_test, self.batch_nr))
        loss = CrossEntropyLoss()
        for test_index in range(n_test):
            for i in range(self.batch_nr):
                indices_sorted = xpl[test_index].argsort(descending=True)
                ds = RestrictedDataset(self.train, indices_sorted[i*self.batchsize:(i+1)*self.batchsize])
                retrained_model = self.retrain(ds)
                self.loss_array[test_index, i] = loss(retrained_model(evalds[start_index + test_index].unsqueeze(0)), evalds_labels[start_index + test_index].unsqueeze(0)).cpu().detach().numpy()
           

    def get_result(self, dir=None, file_name=None):
        # USE THIS WHEN MULTIPLE FILES FOR DIFFERENT XPL ARE READ IN
        #avg_scores = self.scores.mean(dim=0).to('cpu').detach().numpy()
        #self.scores = self.scores.to('cpu').detach().numpy()
        #resdict = {'metric': self.name, 'all_batch_scores': self.scores, 'all_batch_scores_avg': avg_scores,
        #           'scores_for_most_relevant_batch': self.scores[0], 'score_for_most_relevant_batch_avg': avg_scores[0],
        #           'num_batches': self.scores.shape[0]}
        self.scores = self.scores.to('cpu').detach().numpy()
        resdict = {'metric': self.name, 'all_batch_scores': self.loss_array,
                   'num_batches': self.scores.shape}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict

'''
class LinearDatamodelingScore(RetrainMetric):
    name = "LinearDatamodelingScore"

    def __init__(self, train, test, model, num_classes=10, alpha=0.2, samples=10, device="cuda"):
        super().__init__(train, test, model, device)
        self.num_classes = num_classes
        self.alpha = alpha
        self.samples = samples
        self.sample_attributions = np.empty(samples)
        self.sample_accuracies = np.empty(samples)

    def __call__(self, xpl):
        xpl.to(self.device)
        combined_xpl = xpl.sum(dim=0)
        evalds = self.test
        for i in range(self.samples):
            sample_indices = np.random.choice(len(combined_xpl), size= int(self.alpha * len(combined_xpl)), replace=False)
            sample_attribution = combined_xpl[sample_indices].sum()
            ds = RestrictedDataset(self.train, sample_indices)
            retrained_model = self.retrain(ds)
            eval_accuracy = self.evaluate(retrained_model, evalds, num_classes=self.num_classes)

            self.sample_attributions[i] = sample_attribution
            self.sample_accuracies[i] = eval_accuracy
        

    def get_result(self, dir=None, file_name=None):
        spearman = SpearmanCorrCoef()
        resdict = {'metric': self.name, 'sample attributions': self.sample_attributions, 'sample accuracies': self.sample_accuracies,
                   'correlation score': spearman(torch.from_numpy(self.sample_attributions), torch.from_numpy(self.sample_accuracies))}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict
'''
class LinearDatamodelingScore(RetrainMetric):
    name = "LinearDatamodelingScore"

    def __init__(self, dataset_name, train, test, model, epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, alpha=0.05, samples=10, device="cuda"):
        super().__init__(dataset_name, train, test, model,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, device)
        self.alpha = alpha
        self.samples = samples
        self.attribution_array = None
        self.loss_array = None
        self.n_test = None

    def __call__(self, xpl, start_index=0):
        xpl.to(self.device)
        self.n_test = xpl.shape[0]
        evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + xpl.shape[0])], dim=0).to(self.device)
        evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + xpl.shape[0])]).long().to(self.device)
        self.attribution_array = np.empty((xpl.shape[0], self.samples))
        self.loss_array = np.empty((xpl.shape[0], self.samples))
        loss = CrossEntropyLoss()
        for i in range(self.samples):
            sample_indices = np.random.choice(xpl.shape[1], size= int(self.alpha * xpl.shape[1]), replace=False)
            self.attribution_array[:, i] = xpl[:, sample_indices].sum(dim=1).cpu()
            ds = RestrictedDataset(self.train, sample_indices)
            retrained_model = self.retrain(ds)
            self.loss_array[:, i] = loss(retrained_model(evalds), evalds_labels).cpu().detach().numpy()

    def get_result(self, dir=None, file_name=None):
        print(self.attribution_array.shape)
        print(self.loss_array.shape)
        spearman = SpearmanCorrCoef(num_outputs=self.n_test)
        correlation_scores = spearman(torch.from_numpy(self.attribution_array.T), torch.from_numpy(self.loss_array.T))
        resdict = {'metric': self.name, 'correlation_scores': correlation_scores, 'avg_score': correlation_scores.mean(),
                   'sample_attributions': self.attribution_array, 'sample_losses': self.loss_array}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict
        
class LabelFlipMetric(RetrainMetric):
    name = "LabelFlipMetric"

    def __init__(self, dataset_name, train, test, model, epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, batch_nr=10, device="cuda"):
        super().__init__(dataset_name, train, test, model,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, device)
        self.flip_most_relevant=torch.empty(batch_nr+1, dtype=torch.float, device=self.device)
        self.flip_least_relevant=torch.empty(batch_nr+1, dtype=torch.float, device=self.device)
        self.batch_nr = batch_nr
        self.batchsize = len(self.train) // batch_nr

    def __call__(self, xpl):
        xpl.to(self.device)
        combined_xpl = xpl.sum(dim=0)
        indices_sorted = combined_xpl.argsort(descending=True)
        evalds = self.test
        ds_most = FlipLabelDataset(self.train, [])
        ds_least = FlipLabelDataset(self.train, [])
        retrained_model = self.retrain(ds_most)
        eval_accuracy = self.evaluate(retrained_model, evalds, num_classes=self.num_classes)
        self.flip_most_relevant[0] = eval_accuracy
        self.flip_least_relevant[0] = eval_accuracy
        for i in range(self.batch_nr):
            ds_most.corrupt(indices_sorted[i*self.batch_size:(i+1)*self.batch_size])
            ds_least.corrupt(indices_sorted[-(i+1)*self.batch_size:-i*self.batch_size]) if i!= 0 else ds_least.corrupt(indices_sorted[-self.batch_size:]) 
            retrained_model_most = self.retrain(ds_most)
            retrained_model_least = self.retrain(ds_least)
            eval_accuracy_most = self.evaluate(retrained_model_most, evalds, num_classes=self.num_classes)
            eval_accuracy_least = self.evaluate(retrained_model_least, evalds, num_classes=self.num_classes)
            self.flip_most_relevant[i] = eval_accuracy_most
            self.flip_least_relevant[i] = eval_accuracy_least
        

    def get_result(self, dir=None, file_name=None):
        # USE THIS WHEN MULTIPLE FILES FOR DIFFERENT XPL ARE READ IN
        #avg_scores = self.scores.mean(dim=0).to('cpu').detach().numpy()
        #self.scores = self.scores.to('cpu').detach().numpy()
        #resdict = {'metric': self.name, 'all_batch_scores': self.scores, 'all_batch_scores_avg': avg_scores,
        #           'scores_for_most_relevant_batch': self.scores[0], 'score_for_most_relevant_batch_avg': avg_scores[0],
        #           'num_batches': self.scores.shape[0]}
        self.flip_most_relevant = self.flip_most_relevant.to('cpu').detach().numpy()
        self.flip_least_relevant = self.flip_least_relevant.to('cpu').detach().numpy()
        resdict = {'metric': self.name,
                   'accuracies_flip_most_relevant': self.flip_most_relevant,
                   'accuracies_flip_least_relevant': self.flip_least_relevant,
                   'accuracies_delta': self.flip_least_relevant - self.flip_most_relevant}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict