import numpy as np
import librosa
import os

data_dir = 'data'

def get_feature(sample, feature='mfcc', flatten=True):
    y, srate = sample
    if feature == 'mfcc':
        mfccs = librosa.feature.mfcc(y, sr=srate)
        if flatten:
            mfccs = np.mean(mfccs, axis=1)
        return mfccs
    elif feature == 'cqt': 
        cqt = librosa.cqt(y, sr=srate)
        if flatten:
            cqt = np.mean(np.abs(cqt), axis=1)
        return cqt
    elif feature == 'spc':
        spc = librosa.feature.spectral_centroid(y, sr=srate)
        return spc[0]
    
    return None

def lpo_dataset(shuffle=True, limit=None, sample_dur=1, feature='mfcc', feature_flatten=True):
    '''Generates training dataset using monophonic single-note instrument sound samples from 
    the London Philharmonic Orchestra (WAV).'''
    
    samples_dir = f'{data_dir}/lpo_samples'
    instruments = [item for item in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, item))]
    fn_dict = {}

    for instrument in instruments:
        filenames = librosa.util.find_files(f'{samples_dir}/{instrument}', ext=['wav'])
            
        if limit and limit < len(filenames):
            filenames = filenames[0:limit]
            
        fn_dict[instrument] = ['/'.join(fn.split('/')[-3:]) for fn in filenames]
            
    dataset = []
    for label, filenames in fn_dict.items():
        for fn in filenames:
            y, srate = librosa.load(f'{data_dir}/{fn}')
            # zero pad or slice to fit duration
            if len(y) < srate * sample_dur:
                y = np.pad(y, (0, srate * sample_dur - len(y)), 'constant')
            else:
                y = y[0:srate * sample_dur]
                
            sample_feature = get_feature((y, srate), feature=feature, flatten=feature_flatten)
            dataset.append([fn, sample_feature, label])

    if shuffle:
        np.random.shuffle(dataset)
        
    return np.array(dataset)

def get_dataset(name='lpo', feature='mfcc', feature_flatten=True, limit=None):
    if name == 'lpo':
        return lpo_dataset(limit=limit, feature=feature, feature_flatten=feature_flatten)
    
    return None
