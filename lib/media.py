import matplotlib
import matplotlib.pyplot as plt
import numpy as np

colors = [(1, 1, 1), (215/255., 206/255., 223/255.), (141/255., 106/255., 177/255.)]
cm = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', colors, N=3)

white = matplotlib.colors.colorConverter.to_rgba('white',alpha=0.0)
purple = [white, (215/255., 206/255., 223/255.), (141/255., 106/255., 177/255.)]
dark_blue = [white, (155./255., 155./255., 255./255.),(0./255., 0./255., 255./255.)]
green = [white, (155./255., 255./255., 155./255.),(0./255., 127./255., 0./255.)]
red = [white, (255./255., 155./255., 155./255.),(255./255., 0./255., 0./255.)]
light_blue = [white, (155./255., 255./255., 255./255.),(0./255., 204./255., 204./255.)]
pink = [white, (255./255., 155./255., 255./255.),(255./255., 0./255., 255./255.)]
colors_parts  = [purple,dark_blue,green,red,light_blue,pink]
cm_parts = []
for i,cp in enumerate(colors_parts):
    cm_parts.append(matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap_parts{}'.format(i), cp, N=len(colors_parts)))

class PlotFormatter():
    def __init__(self,burnin=1):
        self.plots = []
        self.plot_count = 0
        self.burnin = burnin

    def plot(self,title,vals,color,share=False):
        self.plots.append((title,vals,color,share))
        if not share: self.plot_count += 1

    def show(self,burnin=None):
        if burnin == None: burnin = self.burnin
        rows = self.plot_count//2 + self.plot_count%2
        fig, axes = plt.subplots(rows,2)
        fig.set_figwidth(12)
        fig.set_figheight(5*rows)

        plot_idx = -1
        for i in range(len(self.plots)):
            title, vals, color,share = self.plots[i]
            if not share: plot_idx += 1
            ax = axes[plot_idx//2,plot_idx%2]
            ax.set_title(title)
            ax.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.3)
            x = [k for k in sorted(vals.keys()) if k >= burnin]
            y = [vals[k] for k in x]
            ax.plot(x,y,color=color)

def visualize(x,y=None,buff=5, parts=None):
    x_noparts = np.sum(np.amax(x,axis=1),axis=2).T # ignore parts for now

    if y is not None:
        y_noparts = np.sum(np.amax(y,axis=0),axis=1) # ignore parts for now
        xy = np.hstack([x_noparts,y_noparts[:,None]])
    else:
        xy = x_noparts

    maxnotes = 128-np.argmax(np.flipud(xy>0),axis=0)
    maxnotes = maxnotes[maxnotes<128]
    minnotes = np.argmax(xy>0,axis=0)
    minnotes = minnotes[np.nonzero(minnotes)]

    maxnote = 128 if len(maxnotes) == 0 else np.amax(maxnotes)
    minnote = 0 if len(minnotes) == 0 else np.amin(minnotes)

    # buffer some space at the top/bottom of the plots
    maxnote += buff
    minnote -= buff

    minnote = max(minnote,0) # unlikely but hey

    if y is not None: fig, (ax,ay) = plt.subplots(1,2,figsize=(15,5),sharey=True, gridspec_kw = {'width_ratios':[15, 1]})
    else: fig, ax = plt.subplots(1,1,figsize=(15,5))

    # collapse other parts if they exist
    if parts == None:
         x = np.amax(x,axis=1,keepdims=True)
         y = np.amax(y,axis=0,keepdims=True)
         parts = [0]

    for part in parts:
        ax.imshow(np.sum(x[:,part,minnote:maxnote],axis=2).T,interpolation='none',cmap=cm_parts[part],aspect='auto',
                  norm=matplotlib.colors.Normalize(vmin=0, vmax=2))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda n, pos: '{}'.format(int(n)+minnote)))
        ax.invert_yaxis()

        if y is not None:
            ay.imshow(np.sum(y[part],axis=1).reshape(128,1)[minnote:maxnote],
                      interpolation='none',cmap=cm_parts[part],norm=matplotlib.colors.Normalize(vmin=0, vmax=2),aspect='auto')
            ay.get_xaxis().set_visible(False)
            ay.invert_yaxis()

def sample_to_notes(x):
    note_on = np.zeros(128, dtype=np.int32)
    notes = []

    n = x.shape[0]

    x = np.amax(x,axis=1) # ignore parts for now

    for i in range(n):
        for note in range(128):
            if x[i,note,0] == 0 or x[i,note,1] == 1: # if note is off or newly initiated
                if note_on[note] > 0:                # if it was previously on
                    # close out the note & add its duration to the note list
                    notes.append((i-note_on[note],i,note)) # (onset, offset, note)
                    note_on[note] = 0

            if x[i,note,0] == 1 and note_on[note] > 0: # if note is sustained
                note_on[note] += 1                     # bump its duration counter

            if x[i,note,1] == 1:     # if an onset
                note_on[note] = 1   # initiate the duration counter

    for note in range(128):
        if note_on[note] != 0:
            notes.append((n-note_on[note],n,note))

    return notes,n

def render_notes(notes, n, fs=44100, subbeats=48, tempo=1, shift=0):
    tempo = int((tempo*fs)//2) # samples/beat

    out = np.zeros(int(n*tempo/subbeats), dtype=np.float32)
    for (onset,offset,note) in notes:
        freq = 440.*2**((note + shift - 69.)/12.)
        duration = tempo*(offset-onset)//subbeats
        if duration == 0: continue
        # overtone series!
        mark = 1.00*np.sin(1*freq*2.*np.pi*np.arange(0,duration)/float(fs)) \
             + .250*np.sin(2*freq*2.*np.pi*np.arange(0,duration)/float(fs))
        mark[0:256] *= np.arange(256)/256.
        mark[-256:] *= 1.-np.arange(256)/256.
        out[onset*tempo//subbeats:onset*tempo//subbeats+duration] += mark
    out /= np.max(out)
    return out

