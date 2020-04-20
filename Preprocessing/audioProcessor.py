import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join,split

INPUT_AUDIO = r'../train_FSD/'
OUTPUT_IMG = r'FSDTrain_images/'

class config:
    sampling_rate = 44100
    duration = 5
    samples = sampling_rate * duration
    top_db = 60
    fmin = 20
    fmax =  sampling_rate // 2
    # Spectrogram parameters
    n_mels = 64
    n_fft = 1024
    hop_length = 512
    
def getUnifiedLength(y):
    if len(y) > config.samples: 
        y = y[0:0+config.samples]
    else: 
        y = np.pad(y, (0, config.samples - len(y)), 'wrap')
    return y

def full_frame(width=None, height=None):
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    
def main():
    c = 0
    for f in listdir(INPUT_AUDIO):
        file = join(INPUT_AUDIO,f)
        if not isfile(file):
            continue

        c += 1
        filename = split(file)[1]
        #print("Processing audio file:", filename.split('.')[0])
        x,sr = librosa.load(file, sr=44100)
        x, _ = librosa.effects.trim(x)
        x = getUnifiedLength(x)
        spectrogram = librosa.feature.melspectrogram(x,
                                                    sr=config.sampling_rate,
                                                    n_mels=config.n_mels,
                                                    hop_length=config.hop_length,
                                                    n_fft=config.n_fft,
                                                    fmin=config.fmin,
                                                    fmax=config.fmax)
        logmel = librosa.power_to_db(spectrogram, ref=np.max)
        fig = plt.figure()
        full_frame(10, 7)
        librosa.display.specshow(logmel, sr=config.sampling_rate, hop_length=config.hop_length, x_axis='time', y_axis='mel')
        filename = filename.split('.')[0] + '.png'
        plt.savefig(join(OUTPUT_IMG,filename))
        plt.close(fig)
        plt.close('all')

    print(c)
    
main()
