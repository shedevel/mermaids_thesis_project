import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

''' 
unfinished code towards doing hyper-parameter search for some of the algorithms 
'''


# join features and annotations into one dataframe
def _combine_dataset(features_csv_str, annotations_csv_str):
    features_df = pd.read_csv(features_csv_str, index_col='song_id')
    annotations_df = pd.read_csv(annotations_csv_str, index_col='song_id')
    data_df = features_df.join(annotations_df)
    return data_df.dropna()


# split dimensional dataset into three 2d-arrays: features, valence and arousal
def _split_features_and_dim_annotations(dataset_df):
    valence_nda = dataset_df['valence'].values
    arousal_nda = dataset_df['arousal'].values
    features_nda = dataset_df[[c for c in dataset_df.columns if c not in ['valence', 'arousal', 'quadrant']]].values
    return features_nda, valence_nda, arousal_nda


# takes a dataframe and returns two lists
def _get_row_and_column_headers(dataset_df):
    rows = dataset_df.index.values.tolist()
    cols = dataset_df[[c for c in dataset_df.columns if c not in ['valence', 'arousal', 'quadrant']]].columns.tolist()
    return rows, cols


# preparing dimensional datasets for transfer, first combine then split, returns dict
def prepare_dimensional_dataset_for_transfer(dataset_dict):
    data_df = _combine_dataset(dataset_dict['feat_csv'], dataset_dict['anno_csv'])
    feat_nda, valence_nda, arousal_nda = _split_features_and_dim_annotations(data_df)
    rows_list, cols_list = _get_row_and_column_headers(data_df)

    return {'name': dataset_dict['name'], 'features': feat_nda, 'valence': valence_nda, 'arousal': arousal_nda,
            'cols': cols_list, 'rows': rows_list}


REGS = {
    'NaiveMean': DummyRegressor(strategy='mean'),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'Ridge': Ridge(),
    'kNN': KNeighborsRegressor(),
    'SVRrbf': SVR(kernel='rbf', gamma='scale'),
    'SVRpoly': SVR(kernel='poly', gamma='scale', degree=1),
    'SVRlin': SVR(kernel='linear', gamma='scale'),
    'DecTree': DecisionTreeRegressor(max_depth=5),
    'RandFor': RandomForestRegressor(n_estimators=10, max_depth=5, max_features=1)
}

pmemo = {
    'name': 'pmemo',
    'extraction': 'librosa',
    'feat_csv': '../data/pmemo/pmemo_features_librosa.csv',
    'anno_csv': '../data/pmemo/pmemo_annotations.csv'
}
pmemo_agg = {
    'name': 'pmemo_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/pmemo/pmemo_agg_features_librosa.csv',
    'anno_csv': '../data/pmemo/pmemo_agg_annotations.csv'
}

deam = {
    'name': 'deam',
    'extraction': 'librosa',
    'feat_csv': '../data/deam/deam_features_librosa.csv',
    'anno_csv': '../data/deam/deam_annotations.csv'
}
deam_agg = {
    'name': 'deam_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/deam/deam_agg_features_librosa.csv',
    'anno_csv': '../data/deam/deam_agg_annotations.csv'
}

datasets = {
    'deam': deam
}

SCALERS = {
    'minmax': MinMaxScaler(),
    'standard': StandardScaler(),
    'robust': RobustScaler()
}

dimensions = ['valence', 'arousal']

w = True
comps = [None, 5, 25, 50]
reg_name = 'SVRrbf'
parameters = {'epsilon': [0.0001, 0.001, 0.01, 0.1], 'C': [0.1, 1, 10, 100, 1000]}

for name in datasets:
    for sca in SCALERS:
        for dim in dimensions:
            for n_comp in comps:
                print('\n' + dim + ' | ' + name + ' | ' + 'PCA: n_comp = ' + str(n_comp) + ' | Whiten = ' + str(w))
                print(str(sca))

                data_dict = prepare_dimensional_dataset_for_transfer(datasets[name])

                scaler = SCALERS[sca]
                features = scaler.fit_transform(data_dict['features'])
                targets = data_dict[dim]

                train_feat, test_feat, train_target, test_target = train_test_split(features, targets,
                                                                                    shuffle=True, test_size=.4,
                                                                                    random_state=42)

                reg = REGS[reg_name]

                gs = GridSearchCV(estimator=reg, param_grid=parameters, cv=5, verbose=1)

                pca = PCA(n_components=n_comp, whiten=w, svd_solver='full')
                train_pc = pca.fit_transform(train_feat)
                gs.fit(X=train_pc, y=train_target)

                print('Best params:')
                print(gs.best_params_)

                print('Best score:')
                print(gs.best_score_)

                print('Testing best model on test-data')
                best_model = gs.best_estimator_
                best_model.fit(train_pc, train_target)

                test_pc = pca.transform(test_feat)
                test_pred = best_model.predict(test_pc)

                r2_val = r2_score(test_target, test_pred)
                print('Test R2: ' + str(r2_val))

                delim = ','
                row_string = delim.join((dim, name, sca, str(n_comp), str(w), reg_name, 'r2', str(gs.best_score_)))
                for param in gs.best_params_:
                    row_string += delim + delim.join((param, str(gs.best_params_[param])))

                row_string += delim + str(r2_val)

                with open('../data/results/gridsearch_' + reg_name + '.csv', 'a') as output:
                    output.write(row_string + '\n')
