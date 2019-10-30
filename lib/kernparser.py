import re
from copy import deepcopy
from collections import defaultdict
import numpy as np

from .instruments import instrument_map

class Spine():
    exclusive_interpretation = None
    frontier = 0 # most recent frontier of this spine
    source = [] # most recent channels of this spine
    
    instrument = None
    staff = None
    rest = False
    
class ParsingError(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)
        
def error(message, line):
    raise ParsingError('[line {}] {}'.format(line+1,message))
    
pitch_map = {'c' : 0, 'd' : 2, 'e' : 4, 'f' : 5, 'g' : 7, 'a' : 9, 'b' : 11}

def parse_humdrum(file,mensural_notation,default_instrument=None):
    """ Parse a humdrum record.
    See http://www.humdrum.org/guide/ch05/
    """
    
    validate_time_signatures = False
    
    event_representation = []
    
    # list of notes for each part
    part_representation = defaultdict(list)
    
    global_frontier = 0
    measure_number = 0
    
    part_overlap = full_overlap = 0
    note_count = 0

    splits = joins = intros = 0

    min_note = 128
    max_note = 0
    
    time_per_measure = None
    
    # because there are more spines than kern spines, we need
    # an index map that maps spine indices to kern spine indices
    channel_map = dict()
    
    spines = [] # state of each spine
    init = True
    pickup = 0 # time before the first bar line
    for lineno,line in enumerate(file):
        #print(line,end='')
        
        linetype = line[0] # either ! for comment, * for interpretion, or other for data
        
        if linetype == '\n': error('blank line',lineno)
        
        if linetype == '!': # comment line
            continue # ignore for now
        
        # if not a comment, then it's tab-separated tokens
        tokens = line.strip('\n').split('\t')
        
        # first non-comment row sets up the spines
        if init:
            init = False
            for i in range(len(tokens)):
                spines.append(Spine())
        
        parts = 0
        for spine in spines:
            if spine.exclusive_interpretation == 'kern': parts += 1
        if parts > 6: error('too many parts ({})'.format(parts), lineno)
        
        #
        # parse a line of interpretation
        #
        if linetype == '*':
            exclusive = False
            for toke in tokens:
                if toke[0] != '*': # if linetype is interpretation, all tokens must be an interpretation
                    error('attempt to mix interpretation and other line type', lineno)
                if len(toke) > 1 and toke[1] == '*': # exclusive spine interpretation
                    exclusive = True
            
            if exclusive:
                for toke in tokens:
                    if toke == '*': continue # null okay
                    if toke[0] != '*' or toke[1] != '*':
                        error('attempt to mix exclusive and tandem interpretations', lineno)
                        
                tokens = [toke[2:] for toke in tokens]
                for i,toke in enumerate(tokens):
                    if toke == '': continue # null
                        
                    if spines[i].exclusive_interpretation:
                        error('attempt to change a previously-defined exclusive interpretation', lineno)
                   
                    spines[i].exclusive_interpretation = toke
                    if toke not in ['kern','dynam','text','silbe','Bnum']:
                        error('unrecognized spine interpretation', lineno)
            else: # tandem spine interpretation
                for toke in tokens:
                    if toke == '*': continue # null is okay
                    elif toke[1] == '*':
                        error('attempt to mix tandem and exclusive interpretations', lineno)
            
                tokens = [toke[1:] for toke in tokens]
                # spine control tokens
                if 'v' in tokens or '^' in tokens or '-' in tokens or '+' in tokens or 'x' in tokens:
                    new_spines = []
                    skip = False
                    for i,toke in enumerate(tokens):
                        if toke == '':
                            new_spines.append(deepcopy(spines[i]))
                            skip = False
                        elif toke == '^':
                            new_spines.append(deepcopy(spines[i]))
                            new_spines.append(deepcopy(spines[i]))
                            skip = False
                            splits += 1
                        elif toke == 'v':
                            if not skip:
                                s = deepcopy(spines[i])
                                s.source.extend(spines[i+1].source)
                                new_spines.append(s)
                            else:
                                if new_spines[-1].exclusive_interpretation != spines[i].exclusive_interpretation:
                                    error('joining spines with inconsistent interpretations', lineno)
                                if spines[i].exclusive_interpretation == 'kern':
                                    if abs(new_spines[-1].frontier - spines[i].frontier) > .0001:
                                        error('joining spines with inconsistent frontiers', lineno)
                            skip = True
                            joins += 1
                        elif toke == '-':
                            skip = False
                        elif toke == '+':
                            new_spines.append(deepcopy(spines[i]))
                            s = Spine()
                            s.frontier = spines[i].frontier # should really be global_frontier??
                            new_spines.append(s)
                            skip = False
                            intros += 1
                        elif toke == 'x':
                            error('spine control \'x\' not yet implemented', lineno)
                        else:
                            error('attempting to combine control spines with other interpretation directives', lineno)
                    spines = new_spines
                # format-specific control tokens
                else:
                    for i,toke in enumerate(tokens):
                        instr = None
                        if toke.startswith('I":'):
                            instr = toke[3:]
                        elif toke.startswith('I"') or toke.startswith('I:'):
                            instr = toke[2:]
                        elif toke.startswith('I'):
                            instr = toke[1:]
                        elif toke.startswith('staff'):
                            spines[i].staff = toke[5:]
                            
                        # if it's a recognized instrument, set it (ignore numbers)
                        if instr != None and re.sub(r'\d+', '', instr).strip() in instrument_map.keys():
                            spines[i].instrument = instr
                    
                    # below is (I believe) valid code for checking time signatures, but there are
                    # some sketchy time signatures in the JRP corpus, so this is turned off for now
                    if validate_time_signatures:
                        # time signature
                        ts = re.search(r'\*M(\d+)/(\d+)(%\d+)?', line)
                        if ts:
                            n,d = int(ts.group(1)),int(ts.group(2))
                            if ts.group(3): d /= int(str(ts.group(3))[1:]) # rational denominator
                            if d == 0: d = .5 # breve
                            tpm = n / d
                            if time_per_measure == None: time_per_measure = tpm # init
                            
                            for i,toke in enumerate(tokens):
                                if spine.exclusive_interpretation != 'kern': continue
                                if toke == '': # null is okay if we're consistent (e.g. introducing new spine)
                                    if tpm != time_per_measure: error('inconsistent time signatures', lineno)
                                    continue
                                
                                ts = re.search(r'(\d+)/(\d+)(%\d+)?', toke)
                                if not ts:
                                    error('invalid time signature syntax: \'{}\''.format(toke), lineno)
                                n,d = int(ts.group(1)),int(ts.group(2))
                                if d == 0: d = .5 # breve
                                if ts.group(3): d /= int(str(ts.group(3))[1:]) # rational denominator
                                toke_tpm = n / d
                                if toke_tpm != tpm: error('inconsistent time signatures', lineno)
                                
                            time_per_measure = tpm
        
        # parse a line of data
        else:
            # compute spine flow from previous event to this one
            new_channel_map = dict()
            flow = np.zeros([6,6])
            channel = 0
            for i,spine in enumerate(spines):
                if spine.exclusive_interpretation != 'kern': continue
                if channel > 5: error('channel overflow', lineno)
                for src in spine.source:
                    if channel_map[src] > 5: error('channel overflow', lineno)
                    flow[channel_map[src],channel] = 1
                spine.source = [i] # this spine is now at index i
                new_channel_map[i] = channel # and index i maps to 'channel'
                channel += 1
            channel_map = new_channel_map
            
            if linetype == '=': # barline
                for toke in tokens:
                    if toke[0] != '=':
                        error('mixed bar-line with other datatype', lineno)
                
                # all frontiers should line up at barlines 
                # (notes can't span the line; if they do they should be written as separate notes with ties)
                frontier = spines[0].frontier
                for spine in spines:
                    if spine.exclusive_interpretation != 'kern': continue
                    if abs(spine.frontier-frontier) > .0001:
                        error('spines out of alignment: {}'.format([s.frontier for s in spines]),lineno)
                
                # dirty check to see if this is the first barline
                # (some scores omit the first bar; we assume these don't have a pickup)
                if line[0:3] == '=1\t': pickup = frontier
                
                # really can't check like this (e.g. 3/8 time)
                # need to bite the bullet & get time signatures working if we want good "location" data
                #adjusted_frontier = 4*(frontier - pickup)
                #if mensural_notation: adjusted_frontier *= 4
                #if abs(adjusted_frontier - round(adjusted_frontier)) > .0001:
                #    print(pickup, frontier, adjusted_frontier)
                #    error('off-beat bar-line', lineno)
                    
                measure_number +=1
                continue # nothing more to process on this record
                
            new_event = np.zeros([6,128]) # 128 midi codes
            new_durations = -1*np.ones([6])# durations: -1 sentinal indicates unused channel, 0 indicates continuation
            
            instrument_notes = defaultdict(int)
            instrument_continuations = defaultdict(int)
            
            frontier = None
            null_event = True
            for i,toke in enumerate(tokens):
                if spines[i].exclusive_interpretation != 'kern': continue # ignore for now
                    
                if spines[i].instrument == None:
                    if default_instrument != None:
                        spines[i].instrument = default_instrument
                    else:
                        error('data record in spine {} with no associated instrument'.format(i),lineno)
                
                instr = spines[i].instrument if spines[i].staff == None else spines[i].instrument + spines[i].staff
                if 'piano' in instr: instr = 'piano' # forget about staves; group all piano notes together
                
                if toke == '.': # null
                    if spines[i].rest == False: instrument_continuations[instr] += 1
                    new_durations[channel_map[i]] = 0 # continuation
                    continue
                
                # we ignore grace notes, which "never share a data record with a regular note
                # http://www.humdrum.org/guide/ch06/
                # (treat it like a null)
                if re.search('(Q|q)',toke):
                    new_durations[channel_map[i]] = 0 # continuation
                    continue
                    
                null_event = False
                
                if frontier:
                    if abs(spines[i].frontier - frontier) > .0001:
                        error('inconsistent frontier',lineno)
                frontier = spines[i].frontier
                
                chord = toke.split(' ')
                
                chord_duration = None
                for item in chord:
                    # duration parsing
                    rational_duration = re.search(r'(\d+)\%(\d+)', item)
                    if rational_duration:
                        numerator = int(rational_duration.group(1))
                        denominator = int(rational_duration.group(2))
                        base_duration = denominator / numerator # not a typo: durations are inverted
                    else:
                        duration = re.search(r'(\d+)',item)
                        if not duration: error('note with unspecified duration', lineno)

                        base_value = duration.group(1)
                        if int(base_value) == 0: # extended duration (breve, longa, maxima, etc.)
                            base_duration = 2**(len(base_value))
                        else: # simple duration: fraction of a whole note (e.g. 4 == quarter note, 8 == eighth note)
                            base_duration = 1/float(base_value)

                    # speed up older scores to modern conventions
                    if mensural_notation: base_duration /= 4

                    dots = item.count('.')

                    duration = base_duration
                    for d in range(dots): duration += base_duration/(2**(d+1))

                    if chord_duration and chord_duration != duration: error('chord specified with inconsistent durations',lineno)
                    chord_duration = duration
                    
                    if 'r' in item: # it's a rest
                        spines[i].rest = True
                        continue # ignore for now
                    else: # better hope it's a note
                        instrument_notes[instr] += 1 
                        note_count += 1
                        spines[i].rest = False
                        note = re.search('([a-gA-G]+)', item)
                        if not note: error('unhandled non-note non-rest item',lineno)
                        note = note.group(1)
                            
                        pitch = note[0].lower()
                        if (pitch == note[0]): # middle C or higher
                            octave = 3 + len(note)
                        else: # below middle C
                            octave = 4 - len(note)
                            
                        sharp = re.search(r'(\#+)', item)
                        flat = re.search(r'(\-+)', item)
                        if sharp and flat: error('note cannot be sharp and flat',lineno)
                        sharp = len(sharp.group(1)) if sharp else 0
                        flat = len(flat.group(1)) if flat else 0

                        midi_code = 60 + 12*(octave-4) + pitch_map[pitch] + sharp - flat
                        
                        min_note = min(min_note,midi_code)
                        max_note = max(max_note,midi_code)
                        new_event[channel_map[i],midi_code] = 1
                        
                part_representation[instr].append((new_event[channel_map[i]],chord_duration,spines[i].frontier))
                        
                spines[i].frontier += chord_duration
                new_durations[channel_map[i]] = chord_duration
            
            # for intruments with a new note, count how many concurrent notes in that instrument's part are chopped
            part_overlap += sum([instrument_continuations[instr] for instr in instrument_notes.keys()])
            full_overlap += sum([instrument_continuations[instr] for instr in instrument_continuations.keys()])
            
            if null_event == False: # event can be null if e.g. event is full of grace notes
                if len(event_representation) == 0: # first event?
                    flow = np.eye(6) # flow is undefined; for modeling convenience, make it the identity
                event_representation.append((new_event,new_durations,flow)) # log the event
            
            if frontier != None:
                for i,toke in enumerate(tokens):
                    if toke == '.' and spines[i].exclusive_interpretation == 'kern': # make sure that nulls are consistent with the frontier
                        if spines[i].frontier <= frontier:
                            error('missing a note in part {}'.format(i),lineno)
            
            global_frontier = max([s.frontier for s in spines])
    
    return event_representation,part_representation,4*global_frontier,4*pickup,part_overlap,full_overlap,note_count,min_note,max_note
