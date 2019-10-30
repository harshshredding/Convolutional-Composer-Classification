import os
import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
       
# Plots the confusion matrix and generates a png file with the plot.
#    
# filename : is the name of the png file you want to generate in the
# model : the model we want to use to generate the confusion matrix
# test_set: the test_set we want to use to generate the confusion matrix
# current folder.
def plot_c_matrix(filename, model, test_set):
    chunk_size = 255
    corpora = test_set.corpora
    dataloader = torch.utils.data.DataLoader(dataset=test_set,batch_size=chunk_size,drop_last=False,shuffle=False)
    c_matrix = np.zeros((len(corpora),len(corpora)))
    for i, (x,y) in enumerate(dataloader):    
        x = model.prepare_data(x)
        yhat = np.argmax(model(x).data.cpu().numpy(),axis=1)
        y = np.argmax(y.numpy(), axis=1)
        print(yhat.shape)
        for i in range(len(yhat)):
            yhat_i = yhat[i]
            y_i = y[i]
            c_matrix[y_i][yhat_i] = c_matrix[y_i][yhat_i] + 1
        
    # Convert values of confusion matrix into percentages
    total_counts = np.sum(c_matrix, 1)
    for i in range(len(corpora)):
        for j in range(len(corpora)):
            if total_counts[i] != 0:
                c_matrix[i][j] = c_matrix[i][j]/total_counts[i]

     # Prepare plot
    composers = np.array(corpora)
    fig, ax = plt.subplots()
    im = ax.imshow(c_matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(composers)))
    ax.set_yticks(np.arange(len(composers)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(composers)
    ax.set_yticklabels(composers)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(composers)):
        for j in range(len(composers)):
            text = ax.text(j, i, '%.5f' % c_matrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion matrix Haydn vs Mozart (Simple Non Linear)")
    fig.set_figheight(40)
    fig.set_figwidth(40)
    fig.tight_layout()
    plt.savefig(filename) # Saves image in the current folder
    plt.show()
