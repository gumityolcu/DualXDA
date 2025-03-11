from utils import Metric
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import os
from utils.data import RestrictedDataset, FlipLabelDataset
from train import load_loss, load_optimizer, load_scheduler, load_augmentation
from evaluate import load_model
from tqdm import tqdm
import numpy as np
from torchmetrics.regression import SpearmanCorrCoef
from torchmetrics.regression import KendallRankCorrCoef

class RetrainMetric(Metric):
    name = "RetrainMetric"
    
    def __init__(self, dataset_name, train, test, model_name,
                 epochs, loss, lr, momentum, optimizer, scheduler,
             weight_decay, augmentation, num_classes, batch_size, device):
        self.dataset_name = dataset_name
        self.train = train
        self.test = test
        self.model_name = model_name #load model WITHOUT checkpoint in evaluate script for this metric!
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.augmentation = augmentation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.num_classes = num_classes
        if dataset_name == "AWA":
            self.num_classes = 50
        else:
            self.num_classes = 10
            
    def retrain(self, ds):
        learning_rates=[]
        train_losses = []

        model = load_model(self.model_name, self.dataset_name, self.num_classes).to(self.device)
        model.train()
        loss = load_loss(self.loss)
        optimizer = load_optimizer(self.optimizer, model, self.lr, self.weight_decay, self.momentum)
        scheduler = load_scheduler(self.scheduler, optimizer, self.epochs)
        train_acc = []
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        #print(optimizer.param_groups)
        #for param_group in optimizer.param_groups:
        #    print(param_group['lr'])
        #print("Loop done.")

        for e in range(self.epochs):
            y_true = torch.empty(0, device=self.device)
            y_out = torch.empty((0, self.num_classes), device=self.device)
            cum_loss = 0
            cnt = 0
            for i, (inputs, targets) in enumerate(tqdm(iter(loader))):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                y_true = torch.cat((y_true, targets), 0)

                optimizer.zero_grad()
                logits = model(inputs)
                l = loss(logits, targets)
                y_out = torch.cat((y_out, logits.detach().clone()), 0)
                l.backward()
                #print(model.classifier.bias)
                optimizer.step()
                #print(model.classifier.bias)
                cum_loss = cum_loss + l
                cnt = cnt + inputs.shape[0]
                #print(y_out)
                #print(f"Batch {i + 1}: Loss = {l.item()}")

                #for name, param in model.named_parameters():
                #    if param.grad is not None:
                #        print(f"Layer: {name} | Gradients: {param.grad.abs().mean()}")
                #    else:
                #        print(f"Layer: {name} has no gradients!")

                #print(y_true)
                #print(y_out)
                #print(l.item())
                #print(optimizer)
            y_pred = torch.argmax(y_out, dim=1)
            #print(y_true)
            #print(y_pred)
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
            current_lr = scheduler.get_lr() if hasattr(scheduler, 'get_lr') else [group['lr'] for group in optimizer.param_groups]
            print(f"Current learning rate: {current_lr}")
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
    
class BatchRetraining(RetrainMetric):
    name = "BatchRetraining"

    def calculate_ce_loss(self, model, dataset, num_classes):
        model.eval()
        num_samples = len(dataset)
        total_loss = 0.
        loss_fct = CrossEntropyLoss(reduction='sum')
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        with torch.no_grad():
            for x, y in data_loader:
                x=x.to(self.device)
                y=y.to(self.device)
                output = model(x)
                loss = loss_fct(output, y)
                total_loss += loss.item()
        avg_loss = total_loss / num_samples
        return avg_loss

    def __init__(self, dataset_name, train, test, model_name, 
                 epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, num_classes, batch_size, batch_nr=10, device="cuda", mode="cum"):
        # mode should be one of "cum", "neg_cum", "leave_batch_out", "single_batch"
        super().__init__(dataset_name, train, test, model_name,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, num_classes, batch_size, device)
        self.batch_nr = batch_nr
        self.batchsize = len(self.train) // batch_nr
        self.mode=mode
        self.loss_array = torch.empty(self.batch_nr)
        self.accuracy = torch.empty(self.batch_nr)


    def __call__(self, xpl, start_index=0):
        xpl.to(self.device)
        xpl = torch.abs(xpl)
        xpl = xpl.sum(dim=0)
        #evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + n_test)], dim=0).to(self.device)
        #evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + n_test)]).long().to(self.device)
        for i in range(self.batch_nr):
            indices_sorted = xpl.argsort(descending=True)
            if self.mode=="cum":
                ds = RestrictedDataset(self.train, indices_sorted[:(i+1)*self.batchsize])
            elif self.mode=="neg_cum":
                ds = RestrictedDataset(self.train, indices_sorted[(self.batch_nr-(i+1))*self.batchsize:])
            elif self.mode=="leave_batch_out":
                ds = RestrictedDataset(self.train, torch.cat((indices_sorted[:i*self.batchsize], indices_sorted[(i+1)*self.batchsize:])))
            elif self.mode=="single_batch":
                ds = RestrictedDataset(self.train, indices_sorted[i*self.batchsize:(i+1)*self.batchsize])
            else:
                raise Exception(f"Unexpected BatchTraining metric mode = {self.mode}")
            retrained_model = self.retrain(ds)
            #new_losses[i]=loss(retrained_model(evalds[start_index + test_index].unsqueeze(0)), evalds_labels[start_index + test_index].unsqueeze(0)).cpu().detach().numpy()
            self.loss_array[i]=self.calculate_ce_loss(retrained_model, self.test, self.num_classes)
            self.accuracy[i]=self.evaluate(retrained_model, self.test, self.num_classes)

    def get_result(self, dir=None, file_name=None):
        # USE THIS WHEN MULTIPLE FILES FOR DIFFERENT XPL ARE READ IN
        #avg_scores = self.scores.mean(dim=0).to('cpu').detach().numpy()
        #self.scores = self.scores.to('cpu').detach().numpy()
        #resdict = {'metric': self.name, 'all_batch_scores': self.scores, 'all_batch_scores_avg': avg_scores,
        #           'scores_for_most_relevant_batch': self.scores[0], 'score_for_most_relevant_batch_avg': avg_scores[0],
        #           'num_batches': self.scores.shape[0]}

        resdict = {'metric': self.name,
                   'all_batch_scores': self.loss_array,
                   'all_batch_accuracies': self.accuracy,
                   'num_batches': self.batch_nr,
                   'mode':self.mode
                   }
        
        if "cum" in self.mode:
            resdict["auc_scores"]=self.loss_array.mean().to('cpu').detach().numpy()
        else:
            kendall = KendallRankCorrCoef(num_outputs=1)
            spearman = SpearmanCorrCoef(num_outputs=1)
            #if self.mode=="single_batch":
                #temp_arr=torch.stack(
                #[
                #    torch.arange(self.batch_nr) for _ in range(self.loss_array.shape[0])
                #],
                #device=self.device
                #)
            #elif self.mode=="leave_batch_out": 
            #    temp_arr=torch.stack(
            #    [
            #        self.batch_nr - torch.arange(self.batch_nr) for _ in range(self.loss_array.shape[0])
            #    ],
            #    device=self.device
            #    ) 
            temp_arr=torch.range(start=1, end=self.batch_nr)
            print(self.loss_array)
            print(temp_arr)

            kendall_scores = kendall(self.loss_array, temp_arr) #rank correlation between how highly the batch is scored and whether it leads to smaller loss
            spearman_scores = spearman(self.loss_array, temp_arr)
            resdict["kendall_scores"]=kendall_scores
            resdict["spearman_scores"]=spearman_scores
            #resdict["spearman_score_avg"]=spearman_scores.mean().to('cpu').numpy()
            #resdict["kendall_scores_avg"]=kendall_scores.mean().to('cpu').numpy()

        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict

# class CumAddBatchInNeg(RetrainMetric):
#     # Add batches in from highest negative relevance to highest positive relevance
#     name = "CumAddBatchInNeg"

#     def __init__(self, dataset_name, train, test, model, epochs, loss, lr, momentum, optimizer, scheduler,
#                  weight_decay, augmentation, batch_nr=10, device="cuda"):
#         super().__init__(dataset_name, train, test, model,
#                          epochs, loss, lr, momentum, optimizer, scheduler,
#                          weight_decay, augmentation, device)
#         self.batch_nr = batch_nr
#         self.batchsize = len(self.train) // batch_nr
#         self.loss_array = None

#     def __call__(self, xpl, start_index=0, n_test=10):
#         xpl.to(self.device)
#         evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + n_test)], dim=0).to(self.device)
#         evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + n_test)]).long().to(self.device)
#         self.loss_array = torch.empty((0, self.batch_nr))
#         loss = CrossEntropyLoss()
#         for test_index in range(n_test):
#             for i in range(self.batch_nr):
#                 indices_sorted = xpl[test_index].argsort(descending=False)
#                 ds = RestrictedDataset(self.train, indices_sorted[:(i+1)*self.batchsize])
#                 retrained_model = self.retrain(ds)
#                 self.loss_array[test_index, i] = loss(retrained_model(evalds[start_index + test_index].unsqueeze(0)), evalds_labels[start_index + test_index].unsqueeze(0)).cpu().detach().numpy()
        

#     def get_result(self, dir=None, file_name=None):
#         # USE THIS WHEN MULTIPLE FILES FOR DIFFERENT XPL ARE READ IN
#         #avg_scores = self.scores.mean(dim=0).to('cpu').detach().numpy()
#         #self.scores = self.scores.to('cpu').detach().numpy()
#         #resdict = {'metric': self.name, 'all_batch_scores': self.scores, 'all_batch_scores_avg': avg_scores,
#         #           'scores_for_most_relevant_batch': self.scores[0], 'score_for_most_relevant_batch_avg': avg_scores[0],
#         #           'num_batches': self.scores.shape[0]}
#         self.scores = self.scores.to('cpu').detach().numpy()
#         resdict = {'metric': self.name, 'all_batch_scores': self.loss_array,
#                    'num_batches': self.scores.shape}
#         if dir is not None:
#             self.write_result(resdict, dir, file_name)
#         return resdict

# class LeaveBatchOut(RetrainMetric):
#     name = "LeaveBatchOut"

#     def __init__(self, dataset_name, train, test, model, epochs, loss, lr, momentum, optimizer, scheduler,
#                  weight_decay, augmentation, batch_nr=10, device="cuda"):
#         super().__init__(dataset_name, train, test, model,
#                          epochs, loss, lr, momentum, optimizer, scheduler,
#                          weight_decay, augmentation, device)
#         self.batch_nr = batch_nr
#         self.batchsize = len(self.train) // batch_nr
#         self.loss_array = None

#     def __call__(self, xpl, start_index=0, n_test=10):
#         xpl.to(self.device)
#         evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + n_test)], dim=0).to(self.device)
#         evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + n_test)]).long().to(self.device)
#         self.loss_array = np.empty((n_test, self.batch_nr))
#         loss = CrossEntropyLoss()
#         for test_index in range(n_test):
#             for i in range(self.batch_nr):
#                 indices_sorted = xpl[test_index].argsort(descending=True)
#                 ds = RestrictedDataset(self.train, torch.cat((indices_sorted[:i*self.batchsize], indices_sorted[(i+1)*self.batchsize:])))
#                 retrained_model = self.retrain(ds)
#                 self.loss_array[test_index, i] = loss(retrained_model(evalds[start_index + test_index].unsqueeze(0)), evalds_labels[start_index + test_index].unsqueeze(0)).cpu().detach().numpy()
       

#     def get_result(self, dir=None, file_name=None):
#         # USE THIS WHEN MULTIPLE FILES FOR DIFFERENT XPL ARE READ IN
#         #avg_scores = self.scores.mean(dim=0).to('cpu').detach().numpy()
#         #self.scores = self.scores.to('cpu').detach().numpy()
#         #resdict = {'metric': self.name, 'all_batch_scores': self.scores, 'all_batch_scores_avg': avg_scores,
#         #           'scores_for_most_relevant_batch': self.scores[0], 'score_for_most_relevant_batch_avg': avg_scores[0],
#         #           'num_batches': self.scores.shape[0]}
#         self.scores = self.scores.to('cpu').detach().numpy()
#         resdict = {'metric': self.name, 'all_batch_scores': self.loss_array,
#                    'num_batches': self.scores.shape}
#         if dir is not None:
#             self.write_result(resdict, dir, file_name)
#         return resdict
    
# class OnlyBatch(RetrainMetric):
#     name = "OnlyBatch"

#     def __init__(self, dataset_name, train, test, model, epochs, loss, lr, momentum, optimizer, scheduler,
#                  weight_decay, augmentation, batch_nr=10, device="cuda"):
#         super().__init__(dataset_name, train, test, model,
#                          epochs, loss, lr, momentum, optimizer, scheduler,
#                          weight_decay, augmentation, device)
#         self.batch_nr = batch_nr
#         self.batchsize = len(self.train) // batch_nr
#         self.loss_array = None

#     def __call__(self, xpl, start_index=0, n_test=10):
#         xpl.to(self.device)
#         evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + n_test)], dim=0).to(self.device)
#         evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + n_test)]).long().to(self.device)
#         self.loss_array = np.empty((n_test, self.batch_nr))
#         loss = CrossEntropyLoss()
#         for test_index in range(n_test):
#             for i in range(self.batch_nr):
#                 indices_sorted = xpl[test_index].argsort(descending=True)
#                 ds = RestrictedDataset(self.train, indices_sorted[i*self.batchsize:(i+1)*self.batchsize])
#                 retrained_model = self.retrain(ds)
#                 self.loss_array[test_index, i] = loss(retrained_model(evalds[start_index + test_index].unsqueeze(0)), evalds_labels[start_index + test_index].unsqueeze(0)).cpu().detach().numpy()
           

#     def get_result(self, dir=None, file_name=None):
#         # USE THIS WHEN MULTIPLE FILES FOR DIFFERENT XPL ARE READ IN
#         #avg_scores = self.scores.mean(dim=0).to('cpu').detach().numpy()
#         #self.scores = self.scores.to('cpu').detach().numpy()
#         #resdict = {'metric': self.name, 'all_batch_scores': self.scores, 'all_batch_scores_avg': avg_scores,
#         #           'scores_for_most_relevant_batch': self.scores[0], 'score_for_most_relevant_batch_avg': avg_scores[0],
#         #           'num_batches': self.scores.shape[0]}
#         self.scores = self.scores.to('cpu').detach().numpy()
#         resdict = {'metric': self.name, 'all_batch_scores': self.loss_array,
#                    'num_batches': self.scores.shape}
#         if dir is not None:
#             self.write_result(resdict, dir, file_name)
#         return resdict

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
        combined_xpl = xpl.abs().sum(dim=0)
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

    def __init__(self, dataset_name, train, test, model_name, epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, cache_dir, num_classes, batch_size, alpha=0.5, samples=100, device="cpu"):
        super().__init__(dataset_name, train, test, model_name,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, num_classes, batch_size, device)
        self.alpha = alpha
        self.samples = samples
        self.cache_dir = cache_dir
        self.model_output_array = torch.empty(0,self.samples)
        self.attribution_array = torch.empty(0,self.samples)
        self.n_test = None
        self.device = device

    def __call__(self, xpl, start_index=0):
        xpl=xpl.to(self.device)
        self.n_test = xpl.shape[0]
        evalds = torch.cat([self.test[i][0].unsqueeze(dim=0) for i in range(start_index,start_index + xpl.shape[0])], dim=0).to(self.device)
        evalds_labels = torch.Tensor([self.test[i][1] for i in range(start_index,start_index + xpl.shape[0])]).long().to(self.device)
        attribution_array = torch.empty((xpl.shape[0], self.samples))
        model_output_array = torch.empty((xpl.shape[0], self.samples))
        for i in range(self.samples):
            model_path = os.path.join(self.cache_dir, f'lds{self.alpha}_{i:02d}')
            retrained_model = load_model(self.model_name, self.dataset_name, self.num_classes).to(self.device)
            retrained_model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device))['model_state'])
            retrained_model.eval()
            cur_indices=torch.load(model_path, map_location=torch.device(self.device))['sample_indices']
            attribution_array[:, i] = xpl[:, cur_indices].sum(dim=1).cpu().detach()
            logits = retrained_model(evalds)
            probs = F.softmax(logits, dim=1)
            correct_probs = probs.gather(1, evalds_labels.unsqueeze(1)).squeeze()
            binary_logits = torch.log(correct_probs / (1-correct_probs+1e-10)) #added for numerical stability
            model_output_array[:, i] = binary_logits.cpu().detach()
        self.attribution_array=torch.cat((self.attribution_array, attribution_array), dim=0)
        self.model_output_array=torch.cat((self.model_output_array, model_output_array), dim=0)

    def get_result(self, dir=None, file_name=None):
        spearman = SpearmanCorrCoef(num_outputs=self.n_test)
        correlation_scores = spearman(self.attribution_array.T, self.model_output_array.T)
        resdict = {'metric': self.name, 'correlation_scores': correlation_scores, 'avg_score': correlation_scores.mean(),
                   'sample_attributions': self.attribution_array, 'sample_binarized_logits': self.model_output_array, "subset_indices": self.sample_indices}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict
        
class LabelFlipMetric(RetrainMetric):
    name = "LabelFlipMetric"

    def __init__(self, dataset_name, train, test, model, epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, num_classes, batch_size, batch_nr=10, device="cuda"):
        super().__init__(dataset_name, train, test, model,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, num_classes, batch_size, device)
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