import re
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
file for extracting audio features with the libROSA library
'''


# extract features and summarize each to static features (mean and std over frames) (some will be summarized a lot!)
def extract_highly_aggregated_features(read_from: str, write_to_csv: bool, write_to: str) -> pd.DataFrame:
    feature_names = ['tempo', 'beats_total', 'beats_avg', 'chroma_stft', 'chroma_cq', 'chroma_cens', 'melspectrogram',
                     'mfcc', 'mfcc_delta', 'rms', 'centroid', 'spectral_bw', 'contrast', 'flatness', 'rolloff', 'poly',
                     'tonnetz', 'zcr']

    feature_set = pd.DataFrame()  # feature matrix

    # getting a list of mp3 files in the directory, no recursion
    files = librosa.util.find_files(read_from, 'mp3', False)

    if 'dixon' in read_from:
        pattern = "(\d+)_"
    else:
        pattern = "(\d+).mp3"

    features = dict()
    # traversing all files and extracting features
    for line in files:
        song_id = re.search(pattern, line).group(1)
        print('Extracting features from ' + song_id)

        # loading the audio file to a time-series with default sampling rate of 22050 Hz
        y, sr = librosa.load(line, mono=True)

        # extracting selected features from the libROSA library
        tempo, beats = librosa.beat.beat_track(y, sr)  # high level tempo features
        features['tempo'] = tempo
        features['beats_total'] = beats

        features['chroma_stft'] = librosa.feature.chroma_stft(y, sr)  # chromagram from a waveform or power spectrogram
        features['chroma_cq'] = librosa.feature.chroma_cqt(y, sr)  # constant-Q chromagram
        features['chroma_cens'] = librosa.feature.chroma_cens(y, sr)  # the chroma variant “Chroma Energy Normalized” (CENS)

        features['melspectrogram'] = librosa.feature.melspectrogram(y, sr)  # a mel-scaled spectrogram
        features['mfcc'] = librosa.feature.mfcc(y, sr)  # mel-frequency cepstral coefficients
        features['mfcc_delta'] = librosa.feature.delta(features['mfcc'])

        features['rms'] = librosa.feature.rms(y)  # root mean square energy

        features['centroid'] = librosa.feature.spectral_centroid(y, sr)  # spectral centroid
        features['spectral_bw'] = librosa.feature.spectral_bandwidth(y, sr)  # spectral bandwidth
        features['contrast'] = librosa.feature.spectral_contrast(y, sr)  # spectral contrast
        features['flatness'] = librosa.feature.spectral_flatness(y)  # spectral flatness
        features['rolloff'] = librosa.feature.spectral_rolloff(y, sr)  # spectral roll-off

        features['poly'] = librosa.feature.poly_features(y, sr)  # coefficients of fitting an nth-order polynomial to the columns of a spectrogram
        features['tonnetz'] = librosa.feature.tonnetz(y, sr)  # the tonal centroid features (tonnetz)
        features['zcr'] = librosa.feature.zero_crossing_rate(y)  # zero crossing rate

        # adding mean and stddev values to dataframe
        for name in feature_names:
            if name is 'tempo':
                feature_set.loc[song_id, 'tempo'] = features[name]
            elif name is 'beats_total':
                feature_set.loc[song_id, 'beats_total'] = sum(features[name])
            elif name is 'beats_avg':
                feature_set.loc[song_id, 'beats_avg'] = np.mean(features['total_beats'])
            else:
                feature_set.loc[song_id, name + '_mean'] = np.mean(features[name])
                feature_set.loc[song_id, name + '_std'] = np.std(features[name])

    if write_to_csv:
        feature_set.to_csv(write_to, index_label='song_id')

    return feature_set


# extract features and summarize each to static features (mean and std over frames)
def extract_full_features(read_from: str, write_to_csv: bool, write_to: str, save_images: bool) -> pd.DataFrame:
    feature_set_df = pd.DataFrame()  # feature matrix

    # getting a list of mp3 files in the directory, no recursion
    files = librosa.util.find_files(read_from, 'mp3', False)

    if 'dixon' in read_from:
        pattern = "(\d+)_"
    else:
        pattern = "(\d+).mp3"

    # traversing all files and extracting features
    for line in files:
        song_id = re.search(pattern, line).group(1)

        # reading the file with original sampling rate for info
        y, orig_sr = librosa.load(line, sr=None, mono=True)
        song_length = librosa.core.get_duration(y, orig_sr)

        # only extract features from songs that are less than 90 seconds long
        if song_length <= 90.0:
            print('Extracting features from ' + song_id)

            # resampling to 22050 hz (librosa default)
            sr = 22050
            y = librosa.core.resample(y=y, orig_sr=orig_sr, target_sr=sr)

            if save_images:
                plt.figure(figsize=(14, 5))
                librosa.display.waveplot(y, sr=sr)
                plt.title('Waveplot')
                plt.tight_layout()
                plt.savefig('../data/images/' + song_id + '_waveplot.png')

            tempo, beats = librosa.beat.beat_track(y, sr)  # high level tempo features
            feature_set_df.loc[song_id, 'tempo'] = tempo
            feature_set_df.loc[song_id, 'beats_total'] = sum(beats)
            feature_set_df.loc[song_id, 'beats_avg'] = np.mean(beats)

            chroma_stft = librosa.feature.chroma_stft(y, sr)  # chromagram from a waveform or power spectrogram
            if save_images:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time')
                plt.colorbar()
                plt.title('Chromagram')
                plt.tight_layout()
                plt.savefig('../data/images/' + song_id + '_chromagram_stft.png')

            chroma_stft_dict = mean_and_stddev_on_multidimensional_feature('chroma_stft', chroma_stft)
            for key in chroma_stft_dict:
                feature_set_df.loc[song_id, key] = chroma_stft_dict[key]

            chroma_cens = librosa.feature.chroma_cens(y, sr)  # the chroma variant “Chroma Energy Normalized” (CENS)
            if save_images:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
                plt.colorbar()
                plt.title('Chromagram CENS')
                plt.tight_layout()
                plt.savefig('../data/images/' + song_id + '_chromagram_cens.png')

            chroma_cens_dict = mean_and_stddev_on_multidimensional_feature('chroma_cens', chroma_cens)
            for key in chroma_cens_dict:
                feature_set_df.loc[song_id, key] = chroma_cens_dict[key]

            q_chroma = librosa.feature.chroma_cqt(y, sr)  # constant-Q chromagram
            if save_images:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(q_chroma, y_axis='chroma', x_axis='time')
                plt.colorbar()
                plt.title('Q-Chromagram ')
                plt.tight_layout()
                plt.savefig('../data/images/' + song_id + '_chromagram_q.png')

            q_chroma_dict = mean_and_stddev_on_multidimensional_feature('cons_q_chroma', q_chroma)
            for key in q_chroma_dict:
                feature_set_df.loc[song_id, key] = q_chroma_dict[key]

            melspectrogram = librosa.feature.melspectrogram(y, sr)  # a mel-scaled spectrogram
            if save_images:
                plt.figure(figsize=(10, 4))
                s_dB = librosa.power_to_db(melspectrogram, ref=np.max)
                librosa.display.specshow(s_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel-Frequency Spectrogram')
                plt.tight_layout()
                plt.savefig('../data/images' + song_id + '_melspectrogram.png')

            melspec_dict = mean_and_stddev_on_multidimensional_feature('melspectrogram', melspectrogram)
            for key in melspec_dict:
                feature_set_df.loc[song_id, key] = melspec_dict[key]

            mfcc = librosa.feature.mfcc(y, sr)  # mel-frequency cepstral coefficients
            if save_images:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mfcc, x_axis='time')
                plt.colorbar()
                plt.title('MFCC')
                plt.tight_layout()
                plt.savefig('../data/' + song_id + '_mfcc.png')

            mfcc_dict = mean_and_stddev_on_multidimensional_feature('mfcc', mfcc)
            for key in mfcc_dict:
                feature_set_df.loc[song_id, key] = mfcc_dict[key]

            mfcc_delta = librosa.feature.delta(mfcc)
            if save_images:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mfcc_delta, x_axis='time')
                plt.colorbar()
                plt.title('MFCC Delta')
                plt.tight_layout()
                plt.savefig('../data/images/' + song_id + '_mfcc_delta.png')

            mfcc_delta_dict = mean_and_stddev_on_multidimensional_feature('mfcc_delta', mfcc_delta)
            for key in mfcc_delta_dict:
                feature_set_df.loc[song_id, key] = mfcc_delta_dict[key]

            rms_energy = librosa.feature.rms(y)  # root mean square energy
            feature_set_df.loc[song_id, 'rms_mean'] = np.mean(rms_energy[0])
            feature_set_df.loc[song_id, 'rms_std'] = np.std(rms_energy[0])

            spectral_centroid = librosa.feature.spectral_centroid(y, sr)  # spectral centroid
            feature_set_df.loc[song_id, 'spec_centroid_mean'] = np.mean(spectral_centroid[0])
            feature_set_df.loc[song_id, 'spec_centroid_std'] = np.std(spectral_centroid[0])

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y, sr)  # spectral bandwidth
            feature_set_df.loc[song_id, 'spec_bandwidth_mean'] = np.mean(spectral_bandwidth[0])
            feature_set_df.loc[song_id, 'spec_bandwidth_std'] = np.std(spectral_bandwidth[0])

            spectral_contrast = librosa.feature.spectral_contrast(y, sr)  # spectral contrast
            spectral_contrast_dict = mean_and_stddev_on_multidimensional_feature('spec_contrast', spectral_contrast)
            for key in spectral_contrast_dict:
                feature_set_df.loc[song_id, key] = spectral_contrast_dict[key]

            spectral_flatness = librosa.feature.spectral_flatness(y)  # spectral flatness
            feature_set_df.loc[song_id, 'spec_flatness_mean'] = np.mean(spectral_flatness[0])
            feature_set_df.loc[song_id, 'spec_flatness_std'] = np.std(spectral_flatness[0])

            spectral_rolloff = librosa.feature.spectral_rolloff(y, sr)  # spectral roll-off
            feature_set_df.loc[song_id, 'spec_rolloff_mean'] = np.mean(spectral_rolloff[0])
            feature_set_df.loc[song_id, 'spec_rolloff_std'] = np.std(spectral_rolloff[0])

            # coefficients of fitting an nth-order polynomial to the columns of a spectrogram
            poly_features = librosa.feature.poly_features(y, sr)
            poly_dict = mean_and_stddev_on_multidimensional_feature('poly_feature', poly_features)
            for key in poly_dict:
                feature_set_df.loc[song_id, key] = poly_dict[key]

            tonnetz_features = librosa.feature.tonnetz(y, sr)  # the tonal centroid features (tonnetz)
            tonnetz_dict = mean_and_stddev_on_multidimensional_feature('tonnetz', tonnetz_features)
            for key in tonnetz_dict:
                feature_set_df.loc[song_id, key] = tonnetz_dict[key]

            zcr = librosa.feature.zero_crossing_rate(y)  # zero crossing rate
            feature_set_df.loc[song_id, 'zcr_mean'] = np.mean(zcr[0])
            feature_set_df.loc[song_id, 'zcr_std'] = np.std(zcr[0])

    if write_to_csv:
        feature_set_df.to_csv(write_to, index_label='song_id')

    return feature_set_df


# returns dict
def mean_and_stddev_on_multidimensional_feature(feature_name, features):
    agg_features = {}

    for i in range(len(features)):
        agg_features[feature_name + '_' + str(i + 1) + '_mean'] = np.mean(features[i])
        agg_features[feature_name + '_' + str(i + 1) + '_std'] = np.std(features[i])

    return agg_features

