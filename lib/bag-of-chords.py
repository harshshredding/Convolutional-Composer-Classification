import os
import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

from lib.custom_datasets import FrequencyDataset, DurationDataset, NumOfVoicesDataset, BagOfChordsDataset
from lib.dataset import KernDataset,DatasetSplit

from lib.model import ScoreModel
from lib.opt import optimize
from lib.config import corpora
from lib.bag_of_frequencies import FrequencyWrapper
import lib.media as media
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
checkpoint_dir = '_simple_classifier'
context = 10

train_set = BagOfChordsDataset(context=context, split=DatasetSplit.train)
print(len(train_set))
test_set = BagOfChordsDataset(context=context, split=DatasetSplit.test)
print(len(test_set))

class BagOfChordsModel(ScoreModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def define_graph(self, debug=False):
        hiddenLayerNodes = 100
        self.w1 = Parameter(Tensor(self.m*6, 16))
        self.w2 = Parameter(Tensor(16, hiddenLayerNodes))
        self.w3 = Parameter(Tensor(hiddenLayerNodes, self.composers))
        self.bias = Parameter(Tensor(self.composers))
    
    def forward(self, x):
        res = torch.mm(x[0], self.w1)
        for i in range(1, len(x)):
            res = res + torch.mm(x[i], self.w1)
        res = torch.mm(res, self.w2)
        res = F.relu(res)
        res = torch.mm(res, self.w3)
        res = res + self.bias[None,:].expand(x[0].shape[0],-1)
        return res
        
    def prepare_data(self, x):
        res = ()
        # Build a tuple of variables
        for i in range(len(x)):
            res += (Variable(x[i].cuda(), requires_grad=False),)
        return res

model = BagOfChordsModel(checkpoint_dir, context_length=context, dataset=train_set)
model.initialize()
optimize(model,train_set,test_set,learning_rate=.01)