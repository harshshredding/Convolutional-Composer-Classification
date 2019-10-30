import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

from .base_model import BaseModel
from .config import corpora

class ScoreModel(BaseModel):
    def __init__(self, *args, dataset=None, **kwargs):
        self.composers = len(dataset.corpora)
        self.maxdur = dataset.maxdur+3
        self.offset = dataset.offset
        self.m = dataset.m

        super().__init__(*args, **kwargs)
        self.register_statistic('acc_ts',True,'{:<8.2f}')
        self.register_statistic('acc_tr',True,'{:<8.2f}')
        self.register_statistic('loss_tr',True,'{:<8.2f}')
        self.register_statistic('loss_ts',True,'{:<8.2f}')

    def loss(self, yhat, y):
        return torch.nn.functional.cross_entropy(yhat,Variable(y.cuda()).max(1)[1])*(1/np.log(2))

    def prepare_data(self, x):
        e,t,f,loc = x

        # ship everything over to the gpu
        e = Variable(e.cuda(), requires_grad=False)
        t = Variable(t.cuda(), requires_grad=False)

        return e,t,0,0 # return zeros for backward compatibility

    def compute_stats(self, loader):
        loss = 0
        batch = loader.batch_size
        predictions = []
        ground_truth = []
        for i, (x,y) in enumerate(loader):
            x = self.prepare_data(x)
            yhat = self(x)
            loss += self.loss(yhat,y).data
            predictions.append(np.argmax(yhat.data.cpu().numpy(),axis=1))
            ground_truth.append(np.argmax(y.numpy(),axis=1))
        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)
        loss /= len(loader)

        acc = accuracy_score(ground_truth,predictions)
        return float(loss), 100*acc

    def update_status(self, train_loader, test_loader, last_time, update_time):
        loss, acc = self.compute_stats(test_loader)
        self._tmp_stats['loss_ts'] = loss
        self._tmp_stats['acc_ts'] = acc

        loss, acc = self.compute_stats(train_loader)
        self._tmp_stats['loss_tr'] = loss
        self._tmp_stats['acc_tr'] = acc

        super().update_status(train_loader, test_loader, last_time, update_time)

