from .corpus import parse_raw
from .corpus import find_scores
from .config import corpora as default_corpora
import sys,os,pickle
import numpy as np
from random import shuffle
from lib.opt import optimize
import json
import torch
import matplotlib
import matplotlib.pyplot as plt

# This class performs 10 fold cross validation
# on the given model.
class CrossValidator:
    
    # MODEL is the class of the model
    # we want to perform cross-validation on.
    def __init__(self, MODEL, corpora=default_corpora, is_patches_model = False, patience=25, batch_size=256):
        self.batch_size = batch_size
        self.is_patches_model = is_patches_model
        self.MODEL = MODEL
        self.corpora = corpora
        self.patience = patience
        
    
    # Writes a dataset file which will be useful for replicating
    # a cross validation experiment with a different model.
    # This file contains a datastructure, which is a map from composer names
    # to their corresponding list of scores.
    def write_dataset(self, dataset_file_name = 'cross_dataset.txt'):
        data_dir = 'data'
        data_file = 'voices_data.npz'
        data_path = os.path.join(data_dir,data_file)
        data = dict(np.load(data_path))
        
        # Gather all the scores
        score_ids = []
        for k,v in data.items():
            score_ids.append(k.split(':')[0])
                    
        # A map from composers to their scores.   
        composer_to_scores = {}
            
        # fill up the map
        for composer in self.corpora:
            count =  0
            for score in score_ids:
                if (score.startswith(composer)):
                    composer_scores = composer_to_scores.get(composer, [])
                    composer_scores.append(score)
                    composer_to_scores[composer] = composer_scores
                                        
        # open output file for writing
        with open(dataset_file_name, 'w') as filehandle:  
            json.dump(composer_to_scores, filehandle)
            
    
    def print_composer_accuracies(self, test_set, model):
        dataloader = torch.utils.data.DataLoader(dataset=test_set,batch_size=1,drop_last=False,shuffle=False)
        accuracies = dict()
        for i, (x,y) in enumerate(dataloader):
            x = model.prepare_data(x)
            yhat = np.argmax(model(x).data.cpu().numpy(),axis=1)
            y = np.argmax(y.numpy(),axis=1)
            if self.corpora[y[0]] not in accuracies.keys(): accuracies[self.corpora[y[0]]] = [0,0]
            if self.corpora[y[0]] == self.corpora[yhat[0]]: accuracies[self.corpora[y[0]]][0] += 1
            accuracies[self.corpora[y[0]]][1] += 1

        foo = bar = 0
        for key in sorted(accuracies.keys()):
            print(key,accuracies[key][0],'/',accuracies[key][1],'=',accuracies[key][0]/accuracies[key][1])
            foo += accuracies[key][0]
            bar += accuracies[key][1]
        print(foo,bar,foo/bar)
    
    
    
    # Updates the given confusion matrix with new values based
    # on the current fold. This method is primarily being used to maintaing
    # a running state of the confusion matrix.
    def update_confusion_matrix(self, c_matrix, test_set, model):
        chunk_size = 1
        corpora = self.corpora
        dataloader = torch.utils.data.DataLoader(dataset=test_set,batch_size=chunk_size,drop_last=False,shuffle=False)
        
        
        num_correct = 0
        count = 0
        for i, (x,y) in enumerate(dataloader):    
            x = model.prepare_data(x)
            yhat = np.argmax(model(x).data.cpu().numpy(),axis=1)
            y = np.argmax(y.numpy(), axis=1)  
            for i in range(len(yhat)):
                yhat_i = yhat[i]
                y_i = y[i]
                # If model guessed correctly
                if y_i == yhat_i:
                    num_correct = num_correct + 1
                count = count + 1
                c_matrix[y_i][yhat_i] = c_matrix[y_i][yhat_i] + 1
        print("**************")
        print("Test accuracy : ", num_correct/count)
    
    def plot_confusion_matrix(self, c_matrix):
        # Don't update the orignal copy
        c_matrix = c_matrix.copy()
        # Convert values of confusion matrix into percentages
        total_counts = np.sum(c_matrix, 1)
        for i in range(len(self.corpora)):
            for j in range(len(self.corpora)):
                if total_counts[i] != 0:
                    c_matrix[i][j] = c_matrix[i][j]/total_counts[i]

        # Prepare plot
        composers = np.array(self.corpora)
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
        plt.savefig('cross-validation') # Saves image in the current folder
        plt.show()
    
    
    
    
    # Get validation ids for the ith fold
    def get_validation_ids(self, composer_to_scores, i, num_folds):
        validation_ids = []        
        # get the validation ids for this fold
        for k,v in composer_to_scores.items():
            num_validation_scores = len(v)//num_folds
            if i == (num_folds - 2):
                validation_ids = validation_ids + v[(i+1)*num_validation_scores:len(v)]
            elif i == (num_folds - 1): # Wrap around
                validation_ids = validation_ids + v[0:num_validation_scores]
            else:
                validation_ids = validation_ids + v[(i+1)*num_validation_scores:(i+2)*num_validation_scores]
        return validation_ids
                
    
    
    # Get test ids for the ith fold
    def get_test_ids(self, composer_to_scores, i, num_folds):
        test_ids = []
        # get the test ids for this fold
        for k,v in composer_to_scores.items():
            num_test_scores = len(v)//num_folds
            if (i == (num_folds - 1)):
                test_ids = test_ids + v[i*num_test_scores:len(v)]
            else:
                test_ids = test_ids + v[i*num_test_scores:(i + 1)*num_test_scores]
        return test_ids
    
    
    
    def run(self, context, num_folds=10, dataset_file_name = 'cross_dataset.txt', checkpoint_dir='_simple_classifier'):
        c_matrix = np.zeros((len(self.corpora),len(self.corpora)))
        
        composer_to_scores = {}
        
        # open output file for reading
        with open(dataset_file_name, 'r') as filehandle:  
            composer_to_scores = json.load(filehandle)
    
        # Shuffle the score list of each composer
        for k,v in composer_to_scores.items():
            shuffle(v)   
                     
        fold_data = {}    
        # Do cross validation
        for i in range(num_folds):
            test_ids = self.get_test_ids(composer_to_scores, i, num_folds)                    
            validation_ids = self.get_validation_ids(composer_to_scores, i, num_folds)
            
            print('---------------------------------------------')
            print('---------------------------------------------')
            print('Fold', i)
            # Combine the test and validation ids together.
            test_and_validation_together = validation_ids + test_ids
            
            print('len validation_ids', len(validation_ids))
            print('len test_ids', len(test_ids))
            
            # Get the appropriate dataset type
            if self.is_patches_model:
                from .dataset_custom import KernDataset
                from .dataset_custom import DatasetSplit
            else:
                from .dataset import KernDataset
                from .dataset import DatasetSplit
            
            
            # Make all the datasets
            train_set = KernDataset(context=context,corpora=self.corpora,split=DatasetSplit.train, test_ids=test_and_validation_together, test=True) if (not self.is_patches_model) else KernDataset(context=context,corpora=self.corpora,split=DatasetSplit.train, numpatches = 3, test_ids=test_and_validation_together)
            
            test_set = KernDataset(context=context,corpora=self.corpora,split=DatasetSplit.test, test_ids=test_ids, test=True) if (not self.is_patches_model) else KernDataset(context=context,corpora=self.corpora,split=DatasetSplit.test, numpatches = 3, test_ids=test_ids)
            
            validation_set = KernDataset(context=context,corpora=self.corpora,split=DatasetSplit.test, test_ids=validation_ids, test=True) if (not self.is_patches_model) else KernDataset(context=context,corpora=self.corpora,split=DatasetSplit.test, numpatches = 3 ,test_ids=validation_ids)
                       
            
            model = self.MODEL(checkpoint_dir, context_length=context, dataset=train_set, weight_scale=0.01, avg=.95)
            model.initialize()
                                  
            # Halt when we reach 98% accuracy
            optimize(model, train_set, validation_set, learning_rate=.01, patience=self.patience, batch_size=self.batch_size)
            # Restore checkpoint to the model with best test accuracy
            model.restore_checkpoint(max_folder=True)
            # Update our confusion matrix an plot it
            with model.iterate_averaging():
                self.update_confusion_matrix(c_matrix, test_set, model)
                self.plot_confusion_matrix(c_matrix)
                self.print_composer_accuracies(test_set, model)

        total_scores = np.sum(c_matrix)
        correct = np.trace(c_matrix)
        accuracy = correct / float(total_scores)
        print('Final real overall test accuracy: {} ({}/{})'.format(accuracy, int(correct), int(total_scores)))

        for i, composer in enumerate(self.corpora):
            total_scores = np.sum(c_matrix[i])
            correct = c_matrix[i,i]
            accuracy = correct / float(total_scores)
            print('  {}: {} ({}/{})'.format(composer, accuracy, int(correct), int(total_scores)))
