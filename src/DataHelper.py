import pandas as pd
import sklearn.preprocessing as prep
import os

'''
file for processing and manipulating the datasets
'''


# summarize provided dynamic features to mean and std for each feature
def static_opensmile_features_on_deam(dir_str, write_to):
    mean_of_features = pd.DataFrame()
    std_of_features = pd.DataFrame()

    for dirpath, folders, files in os.walk(dir_str):
        # get columns for the dataframe from the first file in directory
        columns = pd.read_csv(dirpath + '/' + files[0], sep=';', index_col=0).columns
        mean_of_features = mean_of_features.reindex(columns=columns)
        std_of_features = std_of_features.reindex(columns=columns)

        for file in files:
            song_id = file[:-4]
            track_df = pd.read_csv(dirpath + '/' + file, sep=';', index_col=0)

            # compute mean and std over all timestamps for each feature
            means_track_series = track_df.mean(axis=0)
            mean_of_features.loc[song_id] = means_track_series.add_suffix('_mean')

            std_track_series = track_df.std(axis=0)
            std_of_features.loc[song_id] = std_track_series.add_suffix('_std')

    static_features = pd.concat([mean_of_features, std_of_features], axis=1)
    static_features.to_csv(write_to, index_label='song_id')


# create a smaller sample of a dataset with the same percentage-wise representation of genres
def create_smaller_sample_of_dataset(features_csv, metadata_csv, new_size):
    ids_to_keep = []
    meta = pd.read_csv(metadata_csv, index_col=0)
    original_size = len(meta)
    genres = meta['clean_genre'].unique()

    for val in genres:
        subset = meta[meta['clean_genre'] == val]
        subset_sample = _random_sample(subset, original_size, new_size)
        ids_to_keep = ids_to_keep + subset_sample.index.values.tolist()

    features = pd.read_csv(features_csv, index_col=0)
    return features.loc[ids_to_keep]


# get a sample of the subset such that the representation is equal to the original dataset
def _random_sample(subset, original_size, new_size):
    subset_size = len(subset)
    if subset_size is 0:
        return pd.DataFrame()

    sample_size = round((subset_size / original_size) * new_size)
    return subset.sample(sample_size)


# prefixing dixon audio files with id and writing metadata and label csvs
def rename_files_and_write_csv(path_to_audio_files, write_meta_csv, write_label_csv):
    annotations = pd.DataFrame(columns=['label'])
    metadata = pd.DataFrame(columns=['track'])

    # get all folders in audio, serves as label
    dirs = os.listdir(path_to_audio_files)

    # add song_id to file names, create metadata and annotation csv
    # by doing all at once we ensure that song_id is consistent across filenames and csvs
    song_id = 1
    for folder in dirs:
        folder_path = path_to_audio_files + folder + '/'
        for filename in os.listdir(folder_path):
            new_name = str(song_id) + '_' + filename
            src = folder_path + filename
            dst = folder_path + new_name
            os.rename(src, dst)

            annotations.loc[song_id, 'label'] = folder
            metadata.loc[song_id, 'track'] = filename[:-4]

            song_id += 1

    annotations.to_csv(write_label_csv, index_label='song_id')
    metadata.to_csv(write_meta_csv, index_label='song_id')


# combine features, annotation and assigned labels and write to csv
def write_combined_csv_from_dataset(name, feat_csv, anno_csv):
    dimset_df = _combine_dim_dataset(feat_csv, anno_csv)
    dimset_with_labels_df = _add_labels_to_dim_dataset(dimset_df)

    write_to_str = '../data/' + name + '_features_and_ground_truth.csv'
    dimset_with_labels_df.to_csv(write_to_str, index='song_id')


# combine features and annotations for both dimensions into one dataframe
def _combine_dim_dataset(features_csv_str, annotations_csv_str):
    features_df = pd.read_csv(features_csv_str, index_col=0)
    valence_df = pd.read_csv(annotations_csv_str, index_col=0, usecols=['song_id', 'valence'])
    arousal_df = pd.read_csv(annotations_csv_str, index_col=0, usecols=['song_id', 'arousal'])

    dataset_df = features_df.join(valence_df).join(arousal_df)
    return dataset_df.dropna()


# assign categorical labels to a dimensional dataset according to following rules:
# angry = v <= 0.5, a > 0.5, happy = v > 0.5, a >= 0.5, relax = v >= 0.5, a < 0.5, sad = v < 0.5, a <= 0.5
def _add_labels_to_dim_dataset(dim_dataset_df):
    for song in dim_dataset_df.index.values:
        if dim_dataset_df.at[song, 'valence'] <= 0.5:
            if dim_dataset_df.at[song, 'arousal'] > 0.5:
                dim_dataset_df.at[song, 'label'] = 'angry'
            else:
                dim_dataset_df.at[song, 'label'] = 'sad'
        else:
            if dim_dataset_df.at[song, 'arousal'] >= 0.5:
                dim_dataset_df.at[song, 'label'] = 'happy'
            else:
                dim_dataset_df.at[song, 'label'] = 'relax'

    return dim_dataset_df


# for the transfer experiment with a emotion balanced dataset
def write_balanced_subset_from_dataset(name, features_with_anno_csv):
    data_df = pd.read_csv(features_with_anno_csv, index_col=0)
    emo_balanced_df = _create_emotion_balanced_subset_of_dim_dataset(data_df)

    write_to_str = '../data/' + name + '/' + name + 'bal_features_and_ground_truth.csv'
    emo_balanced_df.to_csv(write_to_str, index_label='song_id')


# for the transfer experiment with a emotion balanced dataset
def _create_emotion_balanced_subset_of_dim_dataset(dataset_df):
    label_values = dataset_df['label'].unique()
    print(label_values)

    subsets_dict = {}
    for value in label_values:
        subset = dataset_df.loc[dataset_df['label'] == value]
        subsets_dict[value] = subset

    sample_size = min([len(x) for x in subsets_dict.values()])
    balanced_df = pd.DataFrame(columns=dataset_df.columns)
    for key in subsets_dict:
        sampled_set_df = subsets_dict[key].sample(sample_size)
        balanced_df = balanced_df.append(sampled_set_df)

    return balanced_df.dropna()


# all experiment code is based on having one csv with features and one with annotations
def write_separate_features_and_annotations_csv_from_combined_dataset(name, dataset_csv):
    dataset = pd.read_csv(dataset_csv, index_col=0)
    features = dataset[dataset.columns[:-3]]
    annotations = dataset[dataset.columns[-3:]]

    features.to_csv('../data/' + name + '_static_features_librosa.csv')
    annotations.to_csv('../data/' + name + '_static_annotations.csv')


# combine two dimensional datasets into one
def merge_dimensional_datasets(a_prefix, a_with_anno_csv, b_prefix, b_with_anno_csv, new_name):
    dataset_a_df = pd.read_csv(a_with_anno_csv)
    dataset_a_df['song_id'] = a_prefix + dataset_a_df['song_id'].astype(str)

    dataset_b_df = pd.read_csv(b_with_anno_csv)
    dataset_b_df['song_id'] = b_prefix + dataset_b_df['song_id'].astype(str)

    merged_datasets_df = dataset_a_df.append(dataset_b_df)
    merged_datasets_df.set_index(keys=['song_id'], inplace=True)

    features_df = merged_datasets_df[merged_datasets_df.columns[:-3]]
    features_df.to_csv('../data/' + new_name + '_static_features_librosa.csv', index='song_id')
    annotations_df = merged_datasets_df[merged_datasets_df.columns[-3:]]
    annotations_df.to_csv('../data/' + new_name + '_static_annotations.csv', index='song_id')


# merge metadata from two datasets into one
def merge_metadata(a_prefix, a_metadata_csv, b_prefix, b_metadata_csv, new_name):
    metadata_a = pd.read_csv(a_metadata_csv)
    metadata_a['song_id'] = a_prefix + metadata_a['song_id'].astype(str)

    metadata_b = pd.read_csv(b_metadata_csv)
    metadata_b['song_id'] = b_prefix + metadata_b['song_id'].astype(str)

    merged_metadata_df = metadata_a.append(metadata_b)
    merged_metadata_df.set_index(keys=['song_id'], inplace=True)

    merged_metadata_df.to_csv('../data/' + new_name + '_metadata.csv', index='song_id')
