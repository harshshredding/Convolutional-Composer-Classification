from lib.dataset import KernDataset
import sys,os,pickle
from collections import defaultdict
from enum import Enum

import numpy as np
from torch.utils.data import Dataset

from . import config
from .corpus import checkout,find_scores,parse_raw

DatasetSplit = Enum('DatasetSplit', 'total train test')
# Returns a histogram of the frequencies for a given frame.
class FrequencyDataset(KernDataset):
	# Initialize a dataset based on the given parameters :
	# split - Whether the dataset should be a test dataset or train dataset or neither (all data).
	# context - The number of events we want to look at, at once.
	# corpora - Which corporas we want included in the dataset
	# pitch_shift - Whether we want to transpose dataset.
	# Shuffle - Whether we would like to shuffle dataset. 
	def __init__(self, split=DatasetSplit.total, context=10, corpora=tuple(config.corpora.keys()), pitch_shift=False, shuffle=False):
		super(FrequencyDataset, self).__init__(split=split, context=context, corpora=corpora, pitch_shift=pitch_shift, shuffle=shuffle)

	def __getitem__(self, index):
		return self.data_to_freq_hist(*self.access(*self.index_to_location(index))) 
	def data_to_freq_hist(self, x, y):
		e,_,_,_ = x
		histogram = np.zeros(KernDataset.get_note_range(self), dtype=np.float32)
		for j in range(e.shape[0]):
			time_frame =  e[j]
			for v in range(len(time_frame)):
				note = time_frame[v]
				noteIndex = np.argmax(note)
				if (note[noteIndex] == 1): # If a note was played
					histogram[noteIndex] = histogram[noteIndex] + 1
		return (histogram, y)

# Returns a histogram of the duration lables seen in a given frame.
class DurationDataset(KernDataset):
	# Initialize a dataset based on the given parameters :
	# split - Whether the dataset should be a test dataset or train dataset or neither (all data).
	# context - The number of events we want to look at, at once.
	# corpora - Which corporas we want included in the dataset
	# pitch_shift - Whether we want to transpose dataset.
	# Shuffle - Whether we would like to shuffle dataset.
	def __init__(self, split=DatasetSplit.total, context=10, corpora=tuple(config.corpora.keys()), pitch_shift=False, shuffle=False):
		super(DurationDataset, self).__init__(split=split, context=context, corpora=corpora, pitch_shift=pitch_shift, shuffle=shuffle)

	def __getitem__(self, index):
		return self.data_to_duration_hist(*self.access(*self.index_to_location(index))) 

	def data_to_duration_hist(self, x, y):
		_, t, _, _ = x
		histogram = np.zeros(len(self.dur_map), dtype=np.float32) # the size is the total number of distinct durs observed.
		for j in range(t.shape[0]):
			time_frame =  t[j]
			for v in range(time_frame.shape[0]):
				dur = time_frame[v]
				durIndex = np.argmax(dur)
				if (dur[durIndex] == 1): # If a duration was indicated
					histogram[durIndex] = histogram[durIndex] + 1
		return (histogram, y)
# Returns the maximum number of voices playing in a given frame.
class NumOfVoicesDataset(KernDataset):
	# Initialize a dataset based on the given parameters :
	# split - Whether the dataset should be a test dataset or train dataset or neither (all data).
	# context - The number of events we want to look at, at once.
	# corpora - Which corporas we want included in the dataset
	# pitch_shift - Whether we want to transpose dataset.
	# Shuffle - Whether we would like to shuffle dataset.
	def __init__(self, split=DatasetSplit.total, context=10, corpora=tuple(config.corpora.keys()), pitch_shift=False, shuffle=False):
		super(NumOfVoicesDataset, self).__init__(split=split, context=context, corpora=corpora, pitch_shift=pitch_shift, shuffle=shuffle)

	def __getitem__(self, index):
		return self.data_to_voice_vector(*self.access(*self.index_to_location(index)))

	def data_to_voice_vector(self, x, y):
		_, t, _, _ = x
		maxVoices = t.shape[1]
		numOfVoices = 0
		for voice in range(maxVoices):
			for frame in range(t.shape[0]):
				if (np.argmax(t[frame][voice]) != 1):
					numOfVoices = numOfVoices + 1
					break
		# Below, I add 1 to the number of voices to handle the zero voices case seperately.
		voice_one_hot = np.zeros(maxVoices + 1, dtype=np.float32)
		voice_one_hot[numOfVoices] = 1
		return (voice_one_hot, y)
# Returns a 2d matrix with dimensions context * (size of all voice event vectors appended together)
class BagOfChordsDataset(KernDataset):
	# Initialize a dataset based on the given parameters :
	# split - Whether the dataset should be a test dataset or train dataset or neither (all data).
	# context - The number of events we want to look at, at once.
	# corpora - Which corporas we want included in the dataset
	# pitch_shift - Whether we want to transpose dataset.
	# Shuffle - Whether we would like to shuffle dataset.
	def __init__(self, split=DatasetSplit.total, context=10, corpora=tuple(config.corpora.keys()), pitch_shift=False, shuffle=False):
		super(BagOfChordsDataset, self).__init__(split=split, context=context, corpora=corpora, pitch_shift=pitch_shift, shuffle=shuffle)

	def __getitem__(self, index):
		return self.data_to_chords_tuple(*self.access(*self.index_to_location(index)))

	def data_to_chords_tuple(self, x, y):
		e, _, _, _ = x
		newEventMatrix = np.zeros((e.shape[0], e.shape[1]*e.shape[2]), dtype=np.float32)
		for i in range(e.shape[0]):
			newEventMatrix[i] = np.concatenate(tuple(e[i]))
		return tuple(newEventMatrix), y# Combine two tuples.