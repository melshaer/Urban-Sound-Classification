import re
import pandas as pd
import numpy as np
from scipy.io import wavfile as wv
from pathlib import Path
import librosa

train_dir = Path('Train/')

# Read in the sound data
data_dic = {}
file_format = 'wav'
train_files = (f for f in train_dir.iterdir())
print(train_files)
for filename in train_files:
    if file_format in str(filename):
        print(filename)
        keyid = re.findall(r'\d+', str(filename))
        rate, data = wv.read(filename)
        data_dic[keyid[0]] = data


def get_features(file_dir):
    files = (f for f in file_dir.iterdir())
    index_list = []
    zc_list = []
    spec_cen_list = []
    spec_roll_list = []
    for f in files:
        if file_format in str(f):
            ind = re.findall(r'\d+', str(f))
            both_ch = data_dic[ind[0]]
            # Append just the left channel if stereo
            if len(both_ch.shape) > 1:
                one_ch = both_ch[:, 0]
            else:
                one_ch = both_ch
            index_list.append(ind[0])
            # Feature 1: Zero Crossing Rate
            zc = librosa.zero_crossings(one_ch, pad=False)
            zc_list.append(sum(zc))
            print('zc computed for {0}'.format(str(f)))
            # Feature 2: Spectral Centroid
            spec_cen = librosa.feature.spectral_centroid(one_ch.astype(float))[0]
            spec_cen_list.append(np.mean(spec_cen))
            print('spec_cen computed for {0}'.format(str(f)))
            # Feature 3: Spectral Roll-off
            spec_roll = librosa.feature.spectral_rolloff(one_ch.astype(float))[0]
            spec_roll_list.append(np.mean(spec_roll))
            print('spec_roll computed for {0}'.format(str(f)))
    return index_list, zc_list, spec_cen_list, spec_roll_list


def get_mfcc_features(file_dir):
    files = (f for f in file_dir.iterdir())
    mfcc_dic = {}
    for f in files:
        if file_format in str(f):
            i = re.findall(r'\d+', str(f))
            both_ch = data_dic[i[0]]
            # Append just the left channel if stereo
            if len(both_ch.shape) > 1:
                one_ch = both_ch[:, 0]
            else:
                one_ch = both_ch
            # Feature 4: Mel-Frequency Cepstral Coefficients (MFCCs)
            mfcc = librosa.feature.mfcc(one_ch.astype(float))
            print(mfcc.shape)
            mfcc_dic[i[0]] = np.mean(mfcc, axis=1)
            print('mfcc computed for {0}'.format(str(f)))
    return mfcc_dic


#ind, zc, spec_cen, spec_roll = get_features(train_dir)
mfcc = get_mfcc_features(train_dir)


'''
# Prepare data frame to store features
col_names = ['ID', 'Zero Crossing Rate', 'Spectral Centroid', 'Spectral Rolloff']
list_cols = [ind, zc, spec_cen, spec_roll]
zipped = zip(col_names, list_cols)
features_df = pd.DataFrame(dict(zipped))
features_df.index = features_df['ID']
features_df.to_csv('features.csv')
'''

# Set up MFCC dataframe
mfcc_df = pd.DataFrame(mfcc).T
coeff = ['coeff_' + str(i) for i in range(20)]
index_df = [i for i in mfcc.keys()]
mfcc_df.columns = coeff
mfcc_df['ID'] = index_df
mfcc_df.set_index('ID')
mfcc_df['ID'] = mfcc_df['ID'].astype(str)

# Set up labels dataframe
labels = pd.read_csv('train.csv')
labels.set_index('ID')
labels['ID'] = labels['ID'].astype(str)
#labels.index.name = 'ID'

# Merge features, labels and mfcc dataframes into dataset_df
features = pd.read_csv('features.csv')
features.set_index('ID')
features['ID'] = features['ID'].astype(str)
df_temp = pd.merge(left=features, right=mfcc_df, on='ID')
dataset_df = pd.merge(left=df_temp, right=labels, on='ID')
dataset_df.to_csv('complete_features_dataset.csv')
