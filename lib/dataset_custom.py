import sys,os,pickle
from collections import defaultdict
from enum import Enum

import numpy as np
from torch.utils.data import Dataset

from . import config
from .corpus import checkout,find_scores,parse_raw

import random

DatasetSplit = Enum('DatasetSplit', 'total train test')

class KernDataset(Dataset):
    """KernScores Voices Dataset.
    """

    data_dir = 'data'
    data_file = 'voices_data.npz'

    NULL = 0
    EMPTY = 1
    START = 2
    
    # Initialize a dataset based on the given parameters :
    # split - Whether the dataset should be a test dataset or train dataset or neither (all data).
    # context - The number of events we want to look at, at once.
    # corpora - Which corporas we want included in the dataset
    # pitch_shift - Whether we want to transpose dataset.
    # Shuffle - Whether we would like to shuffle dataset. 
    def __init__(self, split=DatasetSplit.total, context=10, corpora=tuple(config.corpora.keys()), pitch_shift=False, shuffle=False, numpatches=3, test_ids=config.test_ids):
        self.context = context
        self.corpora = tuple(list(corpora).copy())
        self.shuffle = shuffle
        self.numpatches = numpatches 

        # data augmentation
        self.pitch_shift = pitch_shift

        self.data = self.create_dataset()
        if split == DatasetSplit.train:
            self.data = { k : v for k,v in self.data.items() if k.split(':')[0] not in test_ids }
        elif split == DatasetSplit.test:
            self.data = { k : v for k,v in self.data.items() if k.split(':')[0] in test_ids }
        elif split == DatasetSplit.total: pass
        else: raise ValueError('Invalid DatasetSplit')

        self.data = { k : v for k,v in self.data.items() if k.startswith(self.corpora) }
        self.precompute()

    # Helper function for the constructor
    def precompute(self):
        self.size = 0 # Total number of notes in dataset           
        self.base_idx = dict() 
        self.cumsize = dict()
        self.numOfVoices = dict() # Map from score_id to number of voices
        for score_id,(e,_,_,index) in sorted(self.data.items()):
            self.numOfVoices[score_id] = 0 # counter for the number of voices in the score
            for voice in range(6):
                if len(index[voice]) == 0: continue
                self.numOfVoices[score_id] = self.numOfVoices[score_id] + 1
            self.base_idx[score_id] = self.size
            self.cumsize[self.size] = score_id
            self.size += 1 # skip partial data points at beginning
        self.sorted_base = sorted(self.cumsize.keys())

    def access(self, score_id, i):
        events,durs,flow,index = self.data[score_id]

        # We randomly shift pitch
        ps = 0
        if self.pitch_shift:
            ps = np.random.randint(-6,5)

        corpus = np.zeros(len(self.corpora),dtype=np.int32)
        for i,c in enumerate(self.corpora):
            if score_id.startswith(c):
                corpus[i] = 1
                break
        else: raise KeyError

        e = np.zeros([self.numpatches,self.context,self.max_parts,self.m],dtype=np.float32)
        t = np.zeros([self.numpatches,self.context,self.max_parts,self.maxdur+3],dtype=np.float32)
        pos = np.zeros([self.numpatches,self.max_parts],dtype=np.float32)

        def write_data(j, e, t, pos):
            t_raw = np.full([self.context,self.max_parts],self.START,dtype=np.int32)
            if self.context > j: # need to do some temporal padding
                if ps == 0: e[self.context-j:] = events[0:j].astype(np.float32)
                elif ps > 0: e[self.context-j:,:,ps:] = events[0:j,:,:-ps].astype(np.float32)
                else: e[self.context-j:,:,:ps] = events[0:j,:,-ps:].astype(np.float32)
                t_raw[self.context-j:] = durs[0:j]
            else:
                if ps == 0: e[:] = events[j-self.context:j].astype(np.float32)
                elif ps > 0: e[:,:,ps:] = events[j-self.context:j,:,:-ps].astype(np.float32)
                else: e[:,:,:ps] = events[j-self.context:j,:,-ps:].astype(np.float32)
                t_raw[:] = durs[j-self.context:j].astype(np.int32)
            
            for v in range(self.max_parts):
                for k in range(self.context): t[k, v, t_raw[k,v]] = 1

            pos = np.sum(self.dur_map[durs[:j].astype(np.int32)],axis=0)

            return (e,t,pos),corpus

        if len(events) < self.context:
            patches = [len(events) for _ in range(self.numpatches)]
        elif self.numpatches == 1:
            patches = [random.randint(0, (len(events)-self.context)) + self.context]
        else:
            patches = [k*(len(events)-self.context)//(self.numpatches-1) + self.context for k in range(self.numpatches)]

        for k,j in enumerate(patches):
            write_data(j,e[k],t[k],pos[k]) 

        if self.shuffle:
            o = range(self.max_parts) # original ordering
            p = np.random.permutation(range(1,self.max_parts))
            p = np.insert(p,0,0) # first part is fixed
            e[:,:,o] = e[:,:,p]
            t[:,:,o] = t[:,:,p]
            pos[:,o] = pos[:,p]

        return (e,t,0,pos), corpus # zero for backward compatibility

    def location_to_index(self, score_id, i):
        return self.base_idx[score_id] + i

    def index_to_location(self, index):
        base = self.sorted_base[np.searchsorted(self.sorted_base,index,'right')-1]
        score_id = self.cumsize[base]
        i = index - base
        assert(i == 0)
        return score_id,i

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.access(*self.index_to_location(index))

    def to_raster(self, x, y):
        e,t,pos = x

        e = np.pad(e,[(0,0),(0,0),(self.offset,128-(self.offset+self.m))],'constant')
        t = (48*self.dur_map[np.argmax(t,axis=2)]).astype(np.int32)

        x = np.zeros([np.max(np.sum(t,axis=0)),e.shape[1],e.shape[2],2])
        offset = (48*(max(pos)-pos)).astype(np.int32)
        loc = x.shape[0] - offset
        for i in reversed(range(e.shape[0])):
            for p in range(e.shape[1]):
                if t[i,p] == 0: continue
                if loc[p]-t[i,p] >= 0: x[loc[p]-t[i,p],p,:,1] = e[i,p]
                x[loc[p]-t[i,p]:loc[p],p,:,0] = e[i,p]
                loc[p] -= t[i,p]
        return x

    def get_note_range(self):
        return self.m

    def create_dataset(self):
        data_path = os.path.join(self.data_dir,self.data_file)
        if os.path.isfile(data_path):
            data = dict(np.load(data_path))
            self.dur_map = data.pop('_dur_map')
            self.maxdur = len(self.dur_map)-3
            self.offset = int(data.pop('_min_note'))
            self.m = int(data.pop('_note_range'))
            self.max_parts = int(data.pop('_max_parts'))
            print('Found cached voices datafile at {}'.format(data_path,len(data)))
            return data

        scores_path = os.path.join(self.data_dir,'kernscores')
        checkout(scores_path)
        parsed_scores,min_note,max_note,pickups = parse_raw(find_scores(scores_path),event_rep=True)

        scores_data = dict()
        dur_map = dict()
        
        total_length = 0
        
        for score_id,score in parsed_scores.items():
            pickup = pickups[score_id] % 1
            total_length = total_length + len(score)
        avg_score_len = total_length/len(parsed_scores.items())
        print(len(parsed_scores.items()))
        print("avg score len", avg_score_len)
        scores_data = dict()
        dur_map = dict()
        dur_map[0] = self.NULL
        dur_map[-4] = self.EMPTY
        dur_map[-8] = self.START
        m = max_note-min_note+1

        for score_id,score in parsed_scores.items():
            pickup = pickups[score_id] % 1
            events = []
            durs = []
            flow = []
            voice_index = [[] for v in range(6)]
            loc = -48*pickup
            for i in range(len(score)):
                events.append(score[i][0][:,min_note:min_note+m])

                step = 9999
                for v in range(6):
                    dur = 4*score[i][1][v]
                    if not (dur in dur_map): dur_map[dur] = len(dur_map)
                    if dur > 0:
                        step = min(step,dur)
                        voice_index[v].append((i,int(loc)))
                    elif dur == 0: # null
                        assert i > 0 # first event should never be null
                        events[-1][v] = events[-2][v]

                durs.append(np.vectorize(dur_map.__getitem__)(4*score[i][1]))
                flow.append(score[i][2])
        
                loc += 48*step
        
            events = np.stack(events).astype(np.int8)
            durs = np.stack(durs).astype(np.int8)
            flow = np.stack(flow).astype(np.int8)
            scores_data[score_id] = (events,durs,flow,voice_index)

        inv_dur_map = np.zeros(len(dur_map))
        for dur,idx in sorted(dur_map.items()):
            inv_dur_map[idx] = max(dur,0)

        scores_data['_dur_map'] = inv_dur_map
        scores_data['_min_note'] = min_note
        scores_data['_note_range'] = m
        scores_data['_max_parts'] = 6
         
        np.savez(data_path,**scores_data)
    
        self.dur_map = scores_data.pop('_dur_map')
        self.maxdur = len(self.dur_map)-3
        self.offset = int(scores_data.pop('_min_note'))
        self.m = int(scores_data.pop('_note_range'))
        self.max_parts = int(scores_data.pop('_max_parts'))
        return scores_data

