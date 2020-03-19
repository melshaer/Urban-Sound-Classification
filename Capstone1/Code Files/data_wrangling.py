import wave
import re
import pandas as pd
import numpy as np
from scipy.io import wavfile as wv
from pathlib import Path
from readWAV import readwavprop
import librosa

train_dir = Path('Train/')
labels = pd.read_csv('train.csv')
labels.set_index('ID')
labels['ID'] = labels['ID'].astype(str)

# Read in the properties of the wav file and store them in a dataframe
list_names = []
list_rates = []
list_sampwidths = []
list_nchannels = []
list_nframes = []
list_over = []
indexdf = []
train_files = (f for f in train_dir.iterdir())
print(train_files)
for filename in train_files:
    print(filename)
    keyid = re.findall(r'\d+', str(filename))
    try:
        rate, sampwidth, nchannels, nframes = readwavprop(str(filename))
        list_names.append(str(filename))
        list_rates.append(rate)
        list_sampwidths.append(sampwidth)
        list_nframes.append(nframes)
        list_nchannels.append(nchannels)
        indexdf.append(keyid[0])
    except wave.Error:
        print(f"Oversampled file {filename}")
        list_over.append(filename)

col_names = ['sampling rate', 'sampling width in bytes', 'nframes', 'nchannels']
list_cols = [list_rates, list_sampwidths, list_nframes, list_nchannels]
zipped = zip(col_names, list_cols)
dataprop_df = pd.DataFrame(dict(zipped))
dataprop_df.index = indexdf
dataprop_df.to_csv('dataprop.csv')

list_over_df = pd.Series(list_over)
outlist = 'oversampled.csv'
list_over_df.to_csv(outlist)

# Read in the sound data
data_dic = {}
train_files = (f for f in train_dir.iterdir())
print(train_files)
for filename in train_files:
    print(filename)
    keyid = re.findall(r'\d+', str(filename))
    rate, data = wv.read(filename)
    data_dic[keyid[0]] = data


