import os,signal,copy
from time import time
from contextlib import contextmanager

import numpy as np
import torch

def worker_init(args):
    signal.signal(signal.SIGINT, signal.SIG_IGN) # ignore signals so parent can handle them
    np.random.seed(os.getpid() ^ int(time())) # approximately random seed for workers
    
# training_cutoff : The training accuracy (percentage) at which we want to stop training.
# last_iteration : The iteration at which we want to stop training.
# epochs: The number of epochs after which we want to stop training.
# patience: Number of consecutive epochs in which the validation accuracy does not increase and after which we should stop training.
def optimize(model, train_set, test_set, learning_rate=0.01, momentum=.95, batch_size=256, epochs=20000, workers=4, update_rate=1000, l2=0, sample_size=200, last_iteration=0, training_cutoff=101, patience=-1):
    
    maximum_test_accuracy = 0
    epoch_without_increase = 0
    
    kwargs = {'num_workers': workers, 'pin_memory': True, 'worker_init_fn': worker_init}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,**kwargs)

    prng = np.random.RandomState(999)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,**kwargs)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(prng.choice(range(len(train_set)),min(sample_size,len(train_set)),replace=False))
    train_sample_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,sampler=train_sampler,**kwargs)

    print('Initiating optimizer, {} iterations/epoch.'.format(len(train_loader)))
    model.restore_checkpoint()
    print(model.status_header())

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=l2)

    try:
        # This variable represents whether we are done with our training
        finished = False
        t = time()
        for epoch in range(epochs):
            for i, (x,y) in enumerate(train_loader):
                if last_iteration != 0 and i >= last_iteration:
                    break
                if i % update_rate == 0:
                    with model.iterate_averaging():
                        model.update_status(train_sample_loader, test_loader, t, time())
                        model.checkpoint()
                        print(model.status())
                        
                        # If our training accuracy is greater than 98%
                        # we stop training
                        current_training_acc = model.get_current_training_acc()
                                                                       
                        current_test_acc = model.get_current_test_acc()
                        if current_test_acc > maximum_test_accuracy:
                            maximum_test_accuracy = current_test_acc
                            epoch_without_increase = 0
                            # store this model in the max folder
                            model.checkpoint(max_folder = True)
                        
                        # If for a long time the model has not been improving test accuracy
                        # stop training.
                        epoch_without_increase = epoch_without_increase + 1
                        if epoch_without_increase == patience:
                            finished = True
                            break
                        
                        if current_training_acc >= training_cutoff:
                            finished = True
                            break
                    t = time()

                optimizer.zero_grad()
                x = model.prepare_data(x)
                loss = model.loss(model(x),y)
                loss.backward()
                optimizer.step()
                model.average_iterates()
            if finished:
                break

    except KeyboardInterrupt:
        print('Graceful Exit')
    else:
        print('Finished')

def terminal_error(model, dataset, batch_size=1000, workers=4):
    kwargs = {'num_workers': workers, 'pin_memory': True, 'worker_init_fn': worker_init}
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,drop_last=True,**kwargs)
    loss, acc = model.compute_stats(dataloader)
    return loss,acc
