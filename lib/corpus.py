import os,errno
from subprocess import call
from collections import defaultdict

import numpy as np

from . import config
from .kernparser import parse_humdrum,ParsingError

def checkout(scores_path):
    for corpus,(subpath,repo) in config.corpora.items():
        path = os.path.join(scores_path,subpath)
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST: raise

        print('Cloning {} to {} ({})'.format(repo,path,corpus))
        call(['git','-C',path,'clone',repo])

def find_scores(scores_path):
    scores = defaultdict(list)
    count = 0
    for corpus,(subpath,repo) in config.corpora.items():
        repo_path = os.path.join(scores_path,subpath,repo.split('/')[-1][:-4])
        for root, dirs, files in os.walk(repo_path):
            for f in files:
                fullpath = os.path.join(root, f)
                if f.endswith('.krn') and not os.path.islink(fullpath):
                    scores[corpus].append(fullpath)
                    count += 1

    print('Found {} scores...'.format(count)) 
    return scores

def parse_raw(scores, event_rep=False):
    parsed_scores = dict()
    pickups = dict()
    min_note = 128
    max_note = 0
    for corpus in config.corpora.keys():
        instr = config.default_instruments[corpus] if corpus in config.default_instruments.keys() else None
        for i,path in enumerate(scores[corpus]):
            score_id = corpus + '.' + os.path.split(path)[-1][:-4]
            if score_id in config.rejected_scores: continue

            with open(path, 'r', encoding='latin-1') as f:
                try:
                    mensural_notation = '/jrp/' in path
                    event_representation,part_representation,_,pickup,_,_,_,mn,mx = \
                        parse_humdrum(f,mensural_notation,default_instrument=instr)
                except ParsingError as e:
                    print('{} : {}'.format(path,str(e)))
                    continue
                except Exception as e:
                    print(path,str(e))
                    raise e

            min_note = min(min_note,mn)
            max_note = max(max_note,mx)
            if event_rep: parsed_scores[score_id] = event_representation
            else: parsed_scores[score_id] = part_representation
            pickups[score_id] = pickup

    return parsed_scores,min_note,max_note,pickups

def parse_parts(scores_path, cache=None):
    if cache != None:
        if os.path.isfile(cache):
            data = dict(np.load(cache))
            min_note = int(data.pop('_min_note'))
            m = int(data.pop('_note_range'))
            dur_map = data.pop('_dur_map')
            print('Found cached parts datafile at {} ({} parts)'.format(cache,len(data)))
            return data,min_note,m,dur_map

    parsed_scores,min_note,max_note,pickups = parse_raw(find_scores(scores_path))

    parts_data = dict()
    dur_map = dict()
    dur_map[0] = 0
    m = max_note-min_note+1
    for score in parsed_scores.keys():
        if corpus in config.piano_corpora: continue # parts dataset doesn't use piano corpora
        pickup = pickups[score] % 1
        for instr in parsed_scores[score].keys():
            previous_location = next_location = 0
            s = sorted(parsed_scores[score][instr],key=lambda x: x[1])
            attach = False # attach new event to a previous event 
            events = [] 
            for event,duration,location in sorted(s,key=lambda x: x[2]):
                # check for overlap
                attach = False
                if location < next_location - .0001: # if it's not at the expected next location
                    if abs(location - previous_location) < .0001: # if it's at the previous location
                        if abs(location + duration - next_location) < .0001: # if it's the same length
                            attach = True # attach it to the previous event
                        
                    if attach == False:
                        if np.sum(event) > 0: # if it's not just a rest
                            #print('skipping at {} overlap in {}'.format(location,score))
                            pass
                        continue
                    
                if location > next_location + .0001:
                    print('gap in {}')
                    raise
                    
                previous_location = location
                next_location = location + duration

                loc = 4*location-pickup # adjust for the pickup
                dur = 4*duration
                if attach:
                    events[-1][:m] = np.clip(events[-1][:m] + event[min_note:min_note+m], 0, 1)
                else:
                    extended_event = np.zeros(m+2)
                    extended_event[:m] = event[min_note:min_note+m]
                    if not (dur in dur_map): dur_map[dur] = len(dur_map)
                    extended_event[m] = dur_map[dur]
                    extended_event[m+1] = round(48*loc) % 48 # making a (for now correct) assumption here that we can rasterize at 48/quarter
                    events.append(extended_event)

            events = np.stack(events).astype(np.int8)
            part_id = score + ':' + instr
            parts_data[part_id] = events

    inv_dur_map = np.zeros(len(dur_map))
    for dur,idx in sorted(dur_map.items()):
        inv_dur_map[idx] = dur

    parts_data['_dur_map'] = inv_dur_map
    parts_data['_min_note'] = min_note
    parts_data['_note_range'] = m

    np.savez(cache,**parts_data)

    dur_map = parts_data.pop('_dur_map')
    min_note = int(parts_data.pop('_min_note'))
    m = int(parts_data.pop('_note_range'))

    return parts_data,min_note,m,dur_map

