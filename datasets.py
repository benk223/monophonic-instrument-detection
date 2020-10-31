import numpy as np
import librosa
import os

def get_features(sample):
    y, srate = sample
    mfcc = librosa.feature.mfcc(y, sr=srate)
    cqt = librosa.cqt(y, sr=srate)
    spc = librosa.feature.spectral_centroid(y, sr=srate)
    
    return [mfcc, cqt, spc]

def lpo_dataset(shuffle=True, limit=None, sample_dur=1):
    '''Generates training dataset using monophonic single-note instrument sound samples from 
    the London Philharmonic Orchestra (WAV).
    '''
    data_dir = 'lpo_samples'
    instruments = [item for item in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, item))]
    sample_dict = {}

    for instrument in instruments:
        filenames = librosa.util.find_files(f'{data_dir}/{instrument}', ext=['wav'])
        if shuffle:
            np.random.shuffle(filenames)
            
        if limit and limit < len(filenames):
            filenames = filenames[0:limit]
            
        samples = []
        for fn in filenames:
            y, srate = librosa.load(fn)
            # zero pad or slice to fit duration
            if len(y) < srate * sample_dur:
                y = np.pad(y, (0, srate * sample_dur - len(y)), 'constant')
            else:
                y = y[0:srate * sample_dur]
            samples.append((y, srate))
                
        sample_dict[instrument] = samples
            
    dataset = []
    for label, samples in sample_dict.items():
        for sample in samples:
            dataset.append([sample, get_features(sample), label])

    return dataset

def get_dataset(name='lpo', limit=None):
    if name == 'lpo':
        return lpo_dataset(limit=limit)
    
    return None
