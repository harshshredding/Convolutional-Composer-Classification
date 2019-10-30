import os,copy,shutil
from time import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

class BaseModel(torch.nn.Module):
    checkpoint_dir = 'checkpoints'

    def __init__(self, checkpoint, init=False, weight_scale=0, avg=.99, context_length=1, offset=0, m=128):
        super().__init__()

        self.cp = os.path.join(self.checkpoint_dir, checkpoint)
        self.cp_max = os.path.join(self.checkpoint_dir, checkpoint + '_max')
        self.context = context_length

        self.stats = dict()
        self._tmp_stats = dict()
        
        self.register_statistic('iter',True,'{:<8}')
        self.iter = 0
        self.stats['iter'][2][self.iter] = self.iter

        self.register_statistic('time',True,'{:<8.2f}')
        self.register_statistic('utime',True,'{:<8.2f}')

        def count_iter(self, in_grad, out_grad):
            self.iter += 1
        self.register_backward_hook(count_iter)

        self.define_graph()
        self.cuda()

        for parm in self.parameters():
            if weight_scale == 0: parm.data.fill_(0)
            else: parm.data.normal_(0, weight_scale)

        self.training=True

        self.avg = avg
        self.averages = copy.deepcopy(list(parm.data for parm in self.parameters()))
        for (name,parm),pavg in zip(self.named_parameters(),self.averages):
            self.register_buffer(name + 'avg', pavg)
            self.register_statistic('|{}|'.format(name),False,format_str='{<8.0f}')

        self.count_parameters(verbose=True)

    @contextmanager
    def iterate_averaging(self):
        self.training=False
        orig_parms = copy.deepcopy(list(parm.data for parm in self.parameters()))
        for parm, pavg in zip(self.parameters(), self.averages):
            parm.data.copy_(pavg)
        yield
        for parm, orig in zip(self.parameters(), orig_parms):
            parm.data.copy_(orig)
        self.training=True

    def initialize(self):
        if os.path.exists(self.cp):
            shutil.rmtree(self.cp)
            shutil.rmtree(self.cp_max)
            os.makedirs(self.cp)
            os.makedirs(self.cp_max)
        else:
            os.makedirs(self.cp)
            os.makedirs(self.cp_max)
        self.checkpoint()
    
    def get_current_training_acc(self):
        iteration_to_accuracy_map = self.stats['acc_tr'][2]
        maximum_iteration = sorted(iteration_to_accuracy_map.keys())[len(iteration_to_accuracy_map) - 1]
        return iteration_to_accuracy_map[maximum_iteration]
    
    def get_current_test_acc(self):
        iteration_to_accuracy_map = self.stats['acc_ts'][2]
        maximum_iteration = sorted(iteration_to_accuracy_map.keys())[len(iteration_to_accuracy_map) - 1]
        return iteration_to_accuracy_map[maximum_iteration]
    

    def register_statistic(self,key,display,format_str):
        self.stats[key] = [display,format_str,dict()]

    def checkpoint(self, max_folder=False):
        if (not max_folder):
            for stat,value in self.stats.items():
                with open(os.path.join(self.cp, stat + '.npy'), 'wb') as f:
                    np.save(f,value[2])
            torch.save(self.state_dict(), os.path.join(self.cp,'checkpoint.pt'))
        else:
            for stat,value in self.stats.items():
                with open(os.path.join(self.cp_max, stat + '.npy'), 'wb') as f:
                    np.save(f,value[2])
            torch.save(self.state_dict(), os.path.join(self.cp_max, 'checkpoint.pt'))

    def restore_checkpoint(self, max_folder=False):
        if (not max_folder):
            for stat in self.stats:
                with open(os.path.join(self.cp, stat + '.npy'), 'rb') as f:
                    self.stats[stat][2] = np.load(f).item()
            self.iter = sorted(self.stats['iter'][2])[-1]
            self.load_state_dict(torch.load(os.path.join(self.cp,'checkpoint.pt'))) 
        else :
            for stat in self.stats:
                with open(os.path.join(self.cp_max, stat + '.npy'), 'rb') as f:
                    self.stats[stat][2] = np.load(f).item()
            self.iter = sorted(self.stats['iter'][2])[-1]
            self.load_state_dict(torch.load(os.path.join(self.cp_max,'checkpoint.pt')))
            
    # call this last if inheriting to make final updates
    def update_status(self, train_loader, test_loader, last_time, update_time):
        self._tmp_stats['iter'] = self.iter

        for name,parm in self.named_parameters():
            if parm.requires_grad:
                self._tmp_stats['|{}|'.format(name)] = parm.norm().data.item()

        self._tmp_stats['time'] = time() - last_time
        self._tmp_stats['utime'] = time() - update_time

        # write everything at once to try to avoid interruption mid-write
        for k,v in self._tmp_stats.items():
            self.stats[k][2][self.iter] = v
        self._tmp_stats = dict()

    def status_header(self):
        return '\t'.join(sorted([key for key,val in self.stats.items() if val[0]]))

    def status(self):
        status_str = ''
        for key,val in sorted(self.stats.items()):
            if val[0]:
                format_str = val[1]
                current_val = val[2][self.iter]
                status_str += format_str.format(current_val)
        return status_str

    def define_graph(self, debug=False):
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError

    def loss(self, yhat, z):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
    
    def average_iterates(self):
        for parm, pavg in zip(self.parameters(), self.averages):
            pavg.mul_(self.avg).add_((1.-self.avg)*parm.data)

    def count_parameters(self, verbose=False):
        count = 0
        for name,parm in self.named_parameters():
            if parm.requires_grad:
                if verbose: print('{} {} ({})'.format(name,parm.shape,np.prod(parm.shape)))
                count += np.prod(parm.shape)

        print('Initialized graph with {} parameters'.format(count))

    def sum_weights(self, group):
        sum_dict = defaultdict(int)
        for stat in self.stats.keys():
            if stat[0] != '|' or stat[-1] != '|': continue
            if('_' in stat): subgroup, name = stat[1:-1].split('_')
            else: subgroup, name = 'shared', stat[1:-1]
            if subgroup == group:
                for k in self.stats[stat][2].keys():
                    sum_dict[k] += self.stats[stat][2][k]
        return sum_dict

