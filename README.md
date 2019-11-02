# Convolutional-Composer-Classification
The machine learning models and the framework used in paper "Convolutional Composer Classification" can be found here.

To run the models you will need python3 and jupyter notebook.
Luckily installing Anaconda installs both, and it also allows us to create python environments! 

We can install anaconda FOR PYTHON 3 using link https://www.anaconda.com/distribution/.

Once anaconda is installed, we can create and enter a python environment like so :
```
$ conda create -n mypy3env python=3 
```

Then we enter/activiate the environment using
```
$ conda activate mypy3env
```

By using a virtual environment, we will make sure that there are no conflicts between the libraries we will use for the project. 

Now, we will start the jupyter notebook with :
```
(mypy3env)$ jupyter notebook
```

When you will run your first model in jupyter notebook, all the music data required for classification will be downloaded to a `./data` folder. This will only happen once; the following runs will used cached data.



## Every Model Ever
All models start with a preamble like the following.
```python
import os
import numpy as np

import torch # torch stuff
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

from lib.dataset_custom import KernDataset,DatasetSplit # library to access music dataset
from lib.model import ScoreModel # base model for every model
from lib.opt import optimize # library to train
from lib.CrossValidation import CrossValidator # the cross validation framework
from lib.config import corpora_for_classification_all_composers as default_corpora
import lib.media as media
```



All models contain a section like the following 
```python
import sys
class MyModel(ScoreModel): # inherit ScoreModel
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def define_graph(self, debug=False): # Define learnable parameters      
        self.w_e = Parameter(Tensor(78, self.composers)) 
        self.w_t = Parameter(Tensor(55, self.composers))
        self.bias = Parameter(Tensor(self.composers))
    
    def forward(self, x): # Define how to iterate
        e,t,_,_ = x  # e represents the note values, t represent the note-duration values
        x_e = e.mean(1).mean(1).mean(1)
        x_t = t.mean(1).mean(1).mean(1)
        return torch.mm(x_e, self.w_e) + torch.mm(x_t, self.w_t) + self.bias[None, :].expand(e.shape[0], -1)
```



