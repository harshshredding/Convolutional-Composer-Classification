import sys,os,pickle
from collections import defaultdict
from enum import Enum

import numpy as np
from torch.utils.data import Dataset

from . import config
from .corpus import checkout,find_scores,parse_raw

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
    def __init__(self, split=DatasetSplit.total, context=10, corpora=tuple(config.corpora.keys()), pitch_shift=False, shuffle=False, test=False, test_ids = config.test_ids):
        self.context = context
        self.corpora = tuple(list(corpora).copy())
        self.shuffle = shuffle

        # data augmentation
        self.pitch_shift = pitch_shift
        
        #print('printing corpora inside dataset', self.corpora)
        
        # if testing, don't randomize
        self.test = test

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
        if self.test: # one data point per score
            self.base_idx = dict() 
            self.cumsize = dict()
            self.size = 0
            for score_id,_ in sorted(self.data.items()):
                self.base_idx[score_id] = self.size
                self.cumsize[self.size] = score_id
                self.size += 1
            self.sorted_base = sorted(self.cumsize.keys())
            return
        
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
            self.size += max(1,len(e)-self.context) # skip partial data points at beginning
        self.sorted_base = sorted(self.cumsize.keys())

    def access(self, score_id, i):
        #print('score id in access', score_id)
        events,durs,flow,index = self.data[score_id]
        if self.test: # always get the same data point if testing
            if len(events) < self.context:
                j = min(i + self.context,len(events))
            else:
                j = (len(events) -  self.context)//2 + self.context
        else:
            j = min(i + self.context,len(events)-1)
        
        # We randomly shift pitch
        ps = 0
        if self.pitch_shift:
            ps = np.random.randint(-6,5)

        e = np.zeros([self.context,self.max_parts,self.m],dtype=np.float32)
        t = np.full([self.context,6],self.START,dtype=np.int32)
        f = np.repeat(np.eye(6,6,dtype=np.float32)[None,:,:],self.context,axis=0)
        if self.context > j: # need to do some temporal padding
            if ps == 0: e[self.context-j:] = events[0:j].astype(np.float32)
            elif ps > 0: e[self.context-j:,:,ps:] = events[0:j,:,:-ps].astype(np.float32)
            else: e[self.context-j:,:,:ps] = events[0:j,:,-ps:].astype(np.float32)
            t[self.context-j:] = durs[0:j]
            f[self.context-j:] = flow[0:j].astype(np.float32) 
        else:
            if ps == 0: e[:] = events[j-self.context:j].astype(np.float32)
            elif ps > 0: e[:,:,ps:] = events[j-self.context:j,:,:-ps].astype(np.float32)
            else: e[:,:,:ps] = events[j-self.context:j,:,-ps:].astype(np.float32)
            t[:] = durs[j-self.context:j].astype(np.int32)
            f[:] = flow[j-self.context:j].astype(np.float32)
        
        t_out = np.zeros([self.context,self.max_parts,self.maxdur+3],dtype=np.float32)
        for v in range(self.max_parts):
            for k in range(self.context): t_out[k, v, t[k,v]] = 1
        t = t_out

        corpus = np.zeros(len(self.corpora),dtype=np.int32)
        for i,c in enumerate(self.corpora):
            #if c == 'haydn.quartets':
            #    print('index of haydn quartets', i)
                
                
            #if c == 'mozart.piano-sonatas':
            #    print('index of piano sonatas', i)
            
            if score_id.startswith(c):
                corpus[i] = 1
                break
        else: raise KeyError

        pos = np.sum(self.dur_map[durs[:j].astype(np.int32)],axis=0)

        if self.shuffle:
            o = range(self.max_parts) # original ordering
            p = np.random.permutation(range(1,self.max_parts))
            p = np.insert(p,0,0) # first part is fixed
            e[:,o] = e[:,p]
            t[:,o] = t[:,p]
            pos[o] = pos[p]
            f[:,o] = f[:,p]; f[:,:,o] = f[:,:,p]

        return (e,t,f,pos),corpus

    def location_to_index(self, score_id, i):
        return self.base_idx[score_id] + i

    def index_to_location(self, index):
        base = self.sorted_base[np.searchsorted(self.sorted_base,index,'right')-1]
        score_id = self.cumsize[base]
        i = index - base
        #print(i)
        return score_id,i

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.access(*self.index_to_location(index))

    def to_raster(self, x, y):
        e,t,f,pos = x

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

    def decode_duration(self,t):
        if np.sum(t) > 1: return 'INVALID'
        if np.sum(t) == 0: return 'O' # masked
        idx = np.argmax(t)
        if idx > 2: out = str(round(self.dur_map[idx],2))
        elif idx == 2: out = '-' # start
        elif idx == 1: out = 'x' # empty
        elif idx == 0: out = '*' # null
        return out

    def decode_notes(self,e):
        out = ''
        for n in range(len(e)):
            if e[n] == 1: out += str(self.offset+n) + ' '
        return out

    def decode_flow(self,f):
        if not np.any(f): return 'NOFLOW'

        out = ''
        for i in range(f.shape[0]):
            for o in range(f.shape[1]):
                if i == o and f[i,o] == 1: # suppress the diagonal
                    continue
            
                if f[i,o] != 0: out += '{}->{},'.format(i,o)
        return out

    def decode_event(self,e,t,f):
        out = ''
        for p in range(len(e)):
            out += self.decode_duration(t[p]) + ' : ' + self.decode_notes(e[p]) + '\t'
    
        out += self.decode_flow(f)
        out += '\n'   
        return out

    def data_to_str(self,x,y):
        e,t,f,pos = x
        out = ''
        for j in range(e.shape[0]):
            out += self.decode_event(e[j],t[j],f[j])

        return out.expandtabs(16)

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
            if (len(score) == 0):
                print(score_id)
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

