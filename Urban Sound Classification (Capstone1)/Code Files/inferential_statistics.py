import re
import pandas as pd
import numpy as np
from scipy.io import wavfile as wv
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statistics import mean
from scipy import stats
from scipy.stats import ttest_ind
import librosa

train_dir = Path('Train/')
labels = pd.read_csv('train.csv')
labels.set_index('ID')
labels['ID'] = labels['ID'].astype(str)

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


# Get specific class files
def get_file(df, cls):
    class_files = df[df['Class'] == cls]['ID']
    return class_files


def get_data(file_index):
    data_list = []
    for ind in file_index:
        both_channels = data_dic[ind]
        # Append just the left channel if stereo
        if len(both_channels.shape) > 1:
            one_channel = both_channels[:, 0]
        else:
            one_channel = both_channels
        data_list.append(one_channel)
    return data_list


# Prep:
dog_bark_files = get_file(labels, 'dog_bark')
jackhammer_files = get_file(labels, 'jackhammer')
air_conditioner_files = get_file(labels, 'air_conditioner')
street_music_files = get_file(labels, 'street_music')
drilling_files = get_file(labels, 'drilling')
children_playing_files = get_file(labels, 'children_playing')
siren_files = get_file(labels, 'siren')
car_horn_files = get_file(labels, 'car_horn')
engine_idling_files = get_file(labels, 'engine_idling')
gun_shot_files = get_file(labels, 'gun_shot')

dog_bark_index = list(dog_bark_files)
jackhammer_index = list(jackhammer_files)
air_conditioner_index = list(air_conditioner_files)
street_music_index = list(street_music_files)
drilling_index = list(drilling_files)
children_playing_index = list(children_playing_files)
siren_index = list(siren_files)
car_horn_index = list(car_horn_files)
engine_idling_index = list(engine_idling_files)
gun_shot_index = list(gun_shot_files)

jackhammer_data_list = get_data(jackhammer_index)
air_conditioner_data_list = get_data(air_conditioner_index)
street_music_data_list = get_data(street_music_index)
siren_data_list = get_data(siren_index)
engine_idling_data_list = get_data(engine_idling_index)
children_playing_data_list = get_data(children_playing_index)
car_horn_data_list = get_data(car_horn_index)
drilling_data_list = get_data(drilling_index)
dog_bark_data_list = get_data(dog_bark_index)
gun_shot_data_list = get_data(gun_shot_index)

# Compute sample of each file (1st dimension reduction)
jackhammer_mean_files = []
for i in jackhammer_data_list:
    jackhammer_mean_files.append(np.mean(i))
# Compute group sample mean (2nd dimension reduction)
jackhammer_mean = mean(jackhammer_mean_files)

# Compute sample of each file (1st dimension reduction)
air_conditioner_mean_files = []
for i in air_conditioner_data_list:
    air_conditioner_mean_files.append(np.mean(i))
# Compute group sample mean (2nd dimension reduction)
air_conditioner_mean = mean(air_conditioner_mean_files)

# Compute sample of each file (1st dimension reduction)
street_music_mean_files = []
for i in street_music_data_list:
    street_music_mean_files.append(np.mean(i))
# Compute group sample mean (2nd dimension reduction)
street_music_mean = mean(street_music_mean_files)

# Claim 1: jackhammer_mean_files + air_conditioner_mean_files + street_music_mean_files
ttest_ind(jackhammer_mean_files, air_conditioner_mean_files)
ttest_ind(jackhammer_mean_files, street_music_mean_files)
ttest_ind(street_music_mean_files, air_conditioner_mean_files)


def adjust_data_len(data_list):
    data_len_list = []
    for file_data in data_list:
        num_samples = file_data.shape[0]
        data_len_list.append(num_samples)
    new_len = min(data_len_list)
    return new_len


# Compute a representative sample for each group/class
# Find least common number of frames first
def find_rep_sample(data_list):
    min_len = adjust_data_len(data_list)
    amp_col = np.empty((len(data_list), min_len))
    for ind, file_data in enumerate(data_list):
        amp_col[ind] = file_data[:min_len]
    rep_sample = np.mean(amp_col, axis=0)
    return rep_sample


def get_zc_rate(file_index):
    zc_list = []
    for ind in file_index:
        both_ch = data_dic[ind]
        # Append just the left channel if stereo
        if len(both_ch.shape) > 1:
            one_ch = both_ch[:, 0]
        else:
            one_ch = both_ch
        zc = librosa.zero_crossings(one_ch, pad=False)
        zc_list.append(sum(zc))
    return zc_list


'''
jackhammer_zc_list = get_zc_rate(jackhammer_index)
air_conditioner_zc_list = get_zc_rate(air_conditioner_index)
street_music_zc_list = get_zc_rate(street_music_index)
siren_zc_list = get_zc_rate(siren_index)
engine_idling_zc_list = get_zc_rate(engine_idling_index)
children_playing_zc_list = get_zc_rate(children_playing_index)
car_horn_zc_list = get_zc_rate(car_horn_index)
drilling_zc_list = get_zc_rate(drilling_index)
dog_bark_zc_list = get_zc_rate(dog_bark_index)
gun_shot_zc_list = get_zc_rate(gun_shot_index)
'''

jackhammer_rep_sample = find_rep_sample(jackhammer_data_list)
air_conditioner_rep_sample = find_rep_sample(air_conditioner_data_list)
street_music_rep_sample = find_rep_sample(street_music_data_list)
siren_rep_sample = find_rep_sample(siren_data_list)
engine_idling_rep_sample = find_rep_sample(engine_idling_data_list)
children_playing_rep_sample = find_rep_sample(children_playing_data_list)
car_horn_rep_sample = find_rep_sample(car_horn_data_list)
drilling_rep_sample = find_rep_sample(drilling_data_list)
dog_bark_rep_sample = find_rep_sample(dog_bark_data_list)
gun_shot_rep_sample = find_rep_sample(gun_shot_data_list)

# What is the correlation between the wave plots of “dog_bark” and “gun_shot”?
com_len0 = min(dog_bark_rep_sample.shape[0], gun_shot_rep_sample.shape[0])
cor_coeff0 = stats.pearsonr(dog_bark_rep_sample[:com_len0], gun_shot_rep_sample[:com_len0])

# What is the correlation between the wave plots of “children_playing” and “street_music”?
com_len1 = min(children_playing_rep_sample.shape[0], street_music_rep_sample.shape[0])
cor_coeff1 = stats.pearsonr(children_playing_rep_sample[:com_len1], street_music_rep_sample[:com_len1])

# What is the correlation between the wave plots of “engine_idling” and “drilling”?
com_len2 = min(engine_idling_rep_sample.shape[0], drilling_rep_sample.shape[0])
cor_coeff2 = stats.pearsonr(engine_idling_rep_sample[:com_len2], drilling_rep_sample[:com_len2])

# Unit Root (Stationary) Tests:
ts0 = adfuller(street_music_rep_sample)
ts1 = adfuller(siren_rep_sample)
ts2 = adfuller(jackhammer_rep_sample)
ts3 = adfuller(air_conditioner_rep_sample)
ts4 = adfuller(engine_idling_rep_sample)
ts5 = adfuller(drilling_rep_sample)
ts6 = adfuller(car_horn_rep_sample)