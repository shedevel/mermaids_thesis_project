from datetime import datetime
import pandas as pd
from math import sqrt
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Lasso, ElasticNet, Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

'''
file for running the preliminary and main experiments
'''


# prepare dataset for regression in one dimension
def prepare_dataset(features_csv, annotations_csv, id_and_dim_cols):
    dataset = combine_features_and_annotations(features_csv, annotations_csv, id_and_dim_cols)
    return _split_into_features_and_targets(dataset)


# combines features and annotations by appending the target column, returns pandas dataframe
def combine_features_and_annotations(features_csv, annotation_csv, id_and_dim_cols):
    features = pd.read_csv(features_csv, index_col=0)
    annotations = pd.read_csv(annotation_csv, index_col=0, usecols=id_and_dim_cols)

    data_df = features.join(annotations)
    return data_df.dropna()


# assumes that the targets are the last column, returns two series array-like objects
def _split_into_features_and_targets(data):
    features = data[data.columns[:-1]].values
    targets = data[data.columns[-1]].values
    return features, targets


# rmse is not a part of the sklearn metrics
def rmse(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))


# 10-fold cross validation regression with supplied parameters
def cross_val_regression(features_list, y_true_list, scaling_str, n_comp, w, regressor):
    scorers = {'rmse': make_scorer(rmse), 'r2': 'r2'}

    reg = make_pipeline(SCALERS[scaling_str], PCA(n_components=n_comp, whiten=w, svd_solver='full'), regressor)
    scores_dict = cross_validate(reg, features_list, y_true_list, scoring=scorers, cv=10, return_train_score=False)

    mean_rmse_score = scores_dict['test_rmse'].mean()
    rmse_std = scores_dict['test_rmse'].std()

    mean_r2 = scores_dict['test_r2'].mean()
    r2_std = scores_dict['test_r2'].std()

    return mean_rmse_score, rmse_std, mean_r2, r2_std


# 10-fold cross validation classification with supplied parameters
def cross_val_classification(features, y_true, scaling_str, n_comp, w, classifier):
    clf = make_pipeline(SCALERS[scaling_str], PCA(n_components=n_comp, whiten=w, svd_solver='full'), classifier)
    scores = cross_validate(clf, features, y_true, cv=10, return_train_score=False)

    mean_acc_score = scores['test_score'].mean()
    mean_std = scores['test_score'].std()

    return mean_acc_score, mean_std


# used for verification of regression code
def replication_experiment(dataset_names):
    delim = ','

    replication = list()
    for name in dataset_names:
        data = DATASETS[name]

        for dim in dimensions_list:
            features, targets = prepare_dataset(data['feat_csv'], data['anno_csv'], ground_truth_cols_dict[dim])

            for reg in REGS:
                rmse_mean, rmse_std, r2_mean, r2_std = cross_val_regression(features, targets, initial_scaling,
                                                                            initial_n_comp, initial_w, REGS[reg])
                rmse_string = dim + delim + name + delim + data['extraction'] + delim + initial_scaling + delim + str(
                    initial_n_comp) + delim + str(initial_w) + delim + reg + delim + 'RMSE' + delim + '{:.3f}'.format(
                    rmse_mean) + delim + '{:.3f}'.format(rmse_std) + delim + date_str
                print(rmse_string)
                replication.append(rmse_string + '\n')

                r2_string = dim + delim + name + delim + data['extraction'] + delim + initial_scaling + delim + str(
                    initial_n_comp) + delim + str(initial_w) + delim + reg + delim + 'R2' + delim + '{:.2f}'.format(
                    r2_mean) + delim + '{:.3f}'.format(r2_std) + delim + date_str
                print(r2_string)
                replication.append(r2_string + '\n')

                if write_to_file:
                    with open(results_dir + 'replication_and_feature_extraction_comparison.csv', 'a') as output:
                        output.write(rmse_string + '\n' + r2_string + '\n')
    return replication


# investigating the effect scaling and pca combinations when given explained variance ratio as parameter to PCA
def scaling_and_reduction_exp_var_experiment(dataset_names):
    scaling_and_reduction = list()
    for name in dataset_names:
        data = DATASETS[name]
        features, targets = prepare_dataset(data['feat_csv'], data['anno_csv'], [0, 1])

        for scaler_name in SCALERS:
            scaler = SCALERS[scaler_name]
            features = scaler.fit_transform(features)

            pca_val = .25
            while pca_val <= .95:
                pca = PCA(pca_val, whiten=True, svd_solver='full')
                pca.fit(features)

                row_string = ','.join((name, data['extraction'], scaler_name, '{:.2f}'.format(pca_val),
                                       str(pca.n_components_), date_str))
                print(row_string)
                scaling_and_reduction.append(row_string + '\n')

                if write_to_file:
                    with open(results_dir + 'scaling_and_reduction.csv', 'a') as output:
                        output.write(row_string + '\n')

                pca_val += .10
    return scaling_and_reduction


# investigating the effect scaling and pca combinations when given n components as parameter to PCA
def scaling_and_reduction_comp_experiment(dataset_names):
    scaling_and_reduction = list()
    for name in dataset_names:
        data = DATASETS[name]
        features, targets = prepare_dataset(data['feat_csv'], data['anno_csv'], [0, 1])

        for scaler_name in SCALERS:
            scaler = SCALERS[scaler_name]
            features = scaler.fit_transform(features)

            # first the none case where n_components == min(n_samples, n_features) - 1
            pca = PCA(None, whiten=True)
            pca.fit(features)

            exp_var = sum(pca.explained_variance_ratio_)
            row_string = ','.join(
                (name, data['extraction'], scaler_name, '{:.2f}'.format(exp_var), str(None), date_str))
            scaling_and_reduction.append(row_string + '\n')

            if write_to_file:
                with open(results_dir + 'scaling_and_reduction.csv', 'a') as output:
                    output.write(row_string + '\n')

            exp_var = .00
            pca_val = 1
            while exp_var < .99:
                pca = PCA(pca_val, whiten=True)
                pca.fit(features)

                exp_var = sum(pca.explained_variance_ratio_)
                row_string = ','.join(
                    (name, data['extraction'], scaler_name, '{:.2f}'.format(exp_var), str(pca_val), date_str))
                scaling_and_reduction.append(row_string + '\n')

                if pca_val == 25:
                    comp_pd = pd.DataFrame(pca.components_)
                    comp_pd['exp_var_ratio'] = pca.explained_variance_ratio_
                    comp_pd.to_csv('../data/results/' + name + '_' + scaler_name + '_' + str(pca_val) + '.csv')

                if write_to_file:
                    with open(results_dir + 'scaling_and_reduction.csv', 'a') as output:
                        output.write(row_string + '\n')

                pca_val += 1
    return scaling_and_reduction


# baseline regression on dimensional datasets before transfer
def baseline_regression(dataset_names):
    baseline = list()
    for name in dataset_names:
        data = DATASETS[name]

        for dim in dimensions_list:
            features, targets = prepare_dataset(data['feat_csv'], data['anno_csv'], ground_truth_cols_dict[dim])

            for reg_name in chosen_regs:
                rmse_mean, rmse_std, r2_mean, r2_std = cross_val_regression(features, targets, chosen_scaling,
                                                                            chosen_n_comp, chosen_w,
                                                                            REGS[reg_name])

                rmse_string = ','.join((dim, name, data['extraction'], chosen_scaling, str(chosen_n_comp),
                                        str(chosen_w),reg_name, 'RMSE', '{:.3f}'.format(rmse_mean),
                                        '{:.3f}'.format(rmse_std), date_str))
                print(rmse_string)
                baseline.append(rmse_string + '\n')

                r2_string = ','.join((dim, name, data['extraction'], chosen_scaling, str(chosen_n_comp), str(chosen_w),
                                      reg_name, 'R2', '{:.2f}'.format(r2_mean), '{:.3f}'.format(r2_std), date_str))
                print(r2_string)
                baseline.append(r2_string + '\n')

                if write_to_file:
                    with open(results_dir + 'baseline.csv', 'a') as output:
                        output.write(rmse_string + '\n' + r2_string + '\n')
    return baseline


# baseline classification on categorical dataset before transfer
def baseline_classification(dataset_names):
    baseline = list()
    for name in dataset_names:
        data = DATASETS[name]
        features, targets = prepare_dataset(data['feat_csv'], data['anno_csv'], ground_truth_cols_dict['label'])

        for clf in CLFS:
            mean_acc, mean_std = cross_val_classification(features, targets, chosen_scaling, chosen_n_comp, chosen_w,
                                                          CLFS[clf])
            row_string = ','.join(('categorical', name, data['extraction'], chosen_scaling, str(chosen_n_comp),
                                   str(chosen_w), clf, 'accuracy', '{:.2f}'.format(mean_acc), '{:.3f}'.format(mean_std),
                                   date_str))
            print(row_string)
            baseline.append(row_string + '\n')

            if write_to_file:
                with open(results_dir + 'baseline.csv', 'a') as output:
                    output.write(row_string + '\n')
    return baseline


# preparing dimensional datasets for transfer, first combine then split, returns dict
def prepare_dimensional_dataset_for_transfer(data_dict):
    data_df = _combine_dataset(data_dict['feat_csv'], data_dict['anno_csv'])
    feat_nda, valence_nda, arousal_nda = _split_features_and_dim_annotations(data_df)
    rows_list, cols_list = _get_row_and_column_headers(data_df)

    return {'name': data_dict['name'], 'features': feat_nda, 'valence': valence_nda, 'arousal': arousal_nda,
            'cols': cols_list, 'rows': rows_list}


# preparing categorical datasets for transfer, first combine then split, returns dict
def prepare_categorical_dataset_for_transfer(data_dict):
    data_df = _combine_dataset(data_dict['feat_csv'], data_dict['anno_csv'])
    feat_nda, lab_nda = _split_features_and_cat_annotations(data_df)
    rows_list, cols_list = _get_row_and_column_headers(data_df)

    return {'name': data_dict['name'], 'features': feat_nda, 'label': lab_nda,
            'cols': cols_list, 'rows': rows_list}


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
    features_nda = dataset_df[[c for c in dataset_df.columns if c not in ['valence', 'arousal', 'label', 'quadrant']]].values
    return features_nda, valence_nda, arousal_nda


# split categorical dataset into two 2d-arrays: features and labels
def _split_features_and_cat_annotations(dataset_df):
    labels_nda = dataset_df['label'].values
    features_nda = dataset_df.drop(columns=['label']).values
    return features_nda, labels_nda


# takes a dataframe and returns two lists
def _get_row_and_column_headers(dataset_df):
    rows = dataset_df.index.values.tolist()
    cols = dataset_df[[c for c in dataset_df.columns if c not in ['valence', 'arousal', 'label']]].columns.tolist()
    return rows, cols


# fit scaling, pca and regressor to first dataset, then predict VA-values on second dataset
def dim_to_dim_transfer_experiment(from_data, to_data):
    train = prepare_dimensional_dataset_for_transfer(from_data)

    # fitting the scaler to the train data features, then transform train features
    scaler = SCALERS[chosen_scaling]
    scaler.fit(train['features'])
    train['features'] = scaler.transform(train['features'])

    # fitting the pca to the training features
    pca = PCA(chosen_n_comp, chosen_w)
    pca.fit(train['features'])
    train_pc = pca.transform(train['features'])

    # transform test features with scaler and then project onto principal components space
    test = prepare_dimensional_dataset_for_transfer(to_data)
    test['features'] = scaler.transform(test['features'])
    test_pc = pca.transform(test['features'])

    delim = ','
    transfer = list()
    for reg_name in chosen_regs:
        ground_truth_and_pred_vals_df = pd.DataFrame(index=test['rows'])
        ground_truth_and_pred_vals_df['dataset'] = test['name']
        ground_truth_and_pred_vals_df['train_set'] = train['name']
        ground_truth_and_pred_vals_df['scaling'] = chosen_scaling
        ground_truth_and_pred_vals_df['pca_val'] = chosen_n_comp

        regressor = REGS[reg_name]
        for dim in dimensions_list:
            # training the regressor on train principal components, testing on test principal components
            regressor.fit(train_pc, train[dim])
            pred_vals_list = regressor.predict(test_pc)

            rmse_score = rmse(test[dim], pred_vals_list)
            row_string = delim.join((
                                    dim, train['name'], test['name'], chosen_scaling, str(chosen_n_comp), str(chosen_w),
                                    reg_name, 'rmse', '{:.3f}'.format(rmse_score), date_str))
            print(row_string)
            transfer.append(row_string)

            r2_value = r2_score(test[dim], pred_vals_list)
            r2_string = delim.join((dim, train['name'], test['name'], chosen_scaling, str(chosen_n_comp), str(chosen_w),
                                    reg_name, 'r2', '{:.2f}'.format(r2_value), date_str))
            print(r2_string)
            transfer.append(r2_string)

            if write_to_file:
                with open(results_dir + 'transfer.csv', 'a') as output:
                    output.write(row_string + '\n' + r2_string + '\n')

            # combine ground truth and predicted vals for writing results to csv
            truth_and_pred_vals_df = combine_annotations_and_predicted_vals(dim, test['rows'], test[dim],
                                                                            pred_vals_list)
            ground_truth_and_pred_vals_df = ground_truth_and_pred_vals_df.join(truth_and_pred_vals_df)

        if write_predictions:
            ground_truth_and_pred_vals_df.to_csv(
                results_dir + train['name'] + '_' + test['name'] + '_' + chosen_scaling + '_' + str(
                    chosen_n_comp) + '_' + str(chosen_w) + '_' + reg_name + '_' + date_str + '.csv',
                index_label='song_id')
    return transfer


# fit scaling, pca and regressor to first dataset, then predict VA-values on second dataset
def dim_to_cat_transfer_experiment(dim_data, cat_data):
    train = prepare_dimensional_dataset_for_transfer(dim_data)
    test = prepare_categorical_dataset_for_transfer(cat_data)

    scaler = SCALERS[chosen_scaling]

    # fitting the scaler to the train data features
    scaler.fit(train['features'])
    train['features'] = scaler.transform(train['features'])
    test['features'] = scaler.transform(test['features'])

    # fitting the pca to the training features
    pca = PCA(chosen_n_comp, chosen_w)
    pca.fit(train['features'])

    # transforming both train and test features with the fitted pca
    train_pc = pca.transform(train['features'])
    test_pc = pca.transform(test['features'])

    transfer = list()
    pred_vals = dict()

    delim = ','
    for reg_name in chosen_regs:
        regressor = REGS[reg_name]
        for dim in dimensions_list:
            # training the regressor on train principal components, testing on test principal components
            regressor.fit(train_pc, train[dim])
            # adding predicted values for current dimension to dict
            pred_vals[dim] = regressor.predict(test_pc)

        truth_and_prediction_df = pd.DataFrame(index=test['rows'])
        truth_and_prediction_df['dataset'] = test['name']
        truth_and_prediction_df['train'] = train['name']
        truth_and_prediction_df['scaling'] = chosen_scaling
        truth_and_prediction_df['pca_val'] = chosen_n_comp
        truth_and_prediction_df = truth_and_prediction_df.join(
            combine_labels_and_VA_predictions(test['rows'], test['label'], pred_vals['valence'], pred_vals['arousal']))

        # calculate accuracy based on true labels and predicted VA quadrant
        accuracy = accuracy_score(test['label'], truth_and_prediction_df['pred_quadrant'])

        row_string = 'categorical' + delim + train['name'] + delim + test[
            'name'] + delim + chosen_scaling + delim + str(
            chosen_n_comp) + delim + str(chosen_w) + delim + reg_name + delim + 'accuracy' + delim + '{:.2f}'.format(
            accuracy) + delim + date_str
        print(row_string)
        transfer.append(row_string + '\n')

        if write_to_file:
            with open(results_dir + 'transfer.csv', 'a') as output:
                output.write(row_string + '\n')

        if write_predictions:
            truth_and_prediction_df.to_csv(
                results_dir + train['name'] + '_' + test['name'] + '_' + chosen_scaling + '_' + str(
                    chosen_n_comp) + '_' + str(chosen_w) + '_' + reg_name + '_' + date_str + '.csv',
                index_label='song_id')
    return transfer


# combine the ground truth values and predicted values for one dimension into one dataframe
def combine_annotations_and_predicted_vals(dimension, ids_list, ground_truth_list, pred_vals_list):
    error_list = list()
    for i in range(len(ground_truth_list)):
        error_list.append(ground_truth_list[i] - pred_vals_list[i])

    ground_truth_ser = pd.Series(ground_truth_list, index=ids_list, name='true_' + dimension)
    pred_vals_ser = pd.Series(pred_vals_list, index=ids_list, name='pred_' + dimension)
    errors_ser = pd.Series(error_list, index=ids_list, name='error_' + dimension)

    combined_df = ground_truth_ser.to_frame().join(pred_vals_ser.to_frame()).join(errors_ser.to_frame())
    return combined_df


# returns dataframe for writing results to csv for further investigation
def combine_labels_and_VA_predictions(rows, labels, pred_valence, pred_arousal):
    labels_series = pd.Series(labels, index=rows, name='label')
    valence_series = pd.Series(pred_valence, index=rows, name='pred_valence')
    arousal_series = pd.Series(pred_arousal, index=rows, name='pred_arousal')
    combined_df = labels_series.to_frame().join(valence_series.to_frame().join(arousal_series.to_frame()))

    combined_df['pred_quadrant'] = [_predicted_VA_quadrant(v, a) for v, a in
                                    zip(combined_df['pred_valence'], combined_df['pred_arousal'])]
    return combined_df


# based on the predicted VA-values, assign a predicted quadrant label such that accuracy can be calculated
def _predicted_VA_quadrant(v, a):
    if (v <= 0.5) & (a > 0.5):
        return 'angry'

    elif (v > 0.5) & (a >= 0.5):
        return 'happy'

    elif (v >= 0.5) & (a < 0.5):
        return 'relax'

    else:
        return 'sad'


REGS = {
    'Dummy': DummyRegressor(strategy='mean'),
    'Lasso': Lasso(),
    'ElaNet': ElasticNet(),
    'Ridge': Ridge(),
    'kNN': KNeighborsRegressor(),
    'SVRrbf': SVR(kernel='rbf', gamma='scale'),
    'SVRpol': SVR(kernel='poly', gamma='scale'),
    'SVRlin': SVR(kernel='linear', gamma='scale'),
    'DecTre': DecisionTreeRegressor(max_depth=5),
    'RanFor': RandomForestRegressor(n_estimators=10, max_depth=5, max_features=1)
}

CLFS = {
    'Dummy': DummyClassifier(strategy='uniform'),
    'Ridge': RidgeClassifier(),
    'kNN': KNeighborsClassifier(),
    'SVMrbf': SVC(kernel='rbf', gamma='scale', decision_function_shape='ovo'),
    'SVMpol': SVC(kernel='poly', gamma='scale', decision_function_shape='ovo'),
    'SVMlin': SVC(kernel='linear', gamma='scale', decision_function_shape='ovo'),
    'DecTre': DecisionTreeClassifier(max_depth=5),
    'RanFor': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
}

SCALERS = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler()
}

pmemo_o = {
    'name': 'pmemo',
    'extraction': 'opensmile',
    'feat_csv': '../data/pmemo/pmemo_features_opensmile.csv',
    'anno_csv': '../data/pmemo/pmemo_annotations.csv'
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
    'feat_csv': '../data/pmemo/agg/pmemo_agg_features_librosa.csv',
    'anno_csv': '../data/pmemo/agg/pmemo_agg_annotations.csv'
}

bal_pmemo = {
    'name': 'bal_pmemo',
    'extraction': 'librosa',
    'feat_csv': '../data/pmemo/bal_pmemo_features_librosa.csv',
    'anno_csv': '../data/pmemo/pmemo_annotations.csv'
}
bal_pmemo_agg = {
    'name': 'bal_pmemo_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/pmemo/agg/bal_pmemo_agg_features_librosa.csv',
    'anno_csv': '../data/pmemo/agg/bal_pmemo_agg_annotations.csv'
}

deam_o = {
    'name': 'deam',
    'extraction': 'opensmile',
    'feat_csv': '../data/deam/deam_features_opensmile.csv',
    'anno_csv': '../data/deam/deam_annotations.csv'
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
    'feat_csv': '../data/deam/agg/deam_agg_features_librosa.csv',
    'anno_csv': '../data/deam/agg/deam_agg_annotations.csv'
}

bal_deam = {
    'name': 'bal_deam',
    'extraction': 'librosa',
    'feat_csv': '../data/deam/bal_deam_features_librosa.csv',
    'anno_csv': '../data/deam/bal_deam_annotations.csv'
}
bal_deam_agg = {
    'name': 'bal_deam_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/deam/agg/bal_deam_agg_features_librosa.csv',
    'anno_csv': '../data/deam/agg/bal_deam_agg_annotations.csv'
}

minideam = {
    'name': 'minideam',
    'extraction': 'librosa',
    'feat_csv': '../data/minideam/minideam_features_librosa.csv',
    'anno_csv': '../data/minideam/minideam_annotations.csv'
}
minideam_agg = {
    'name': 'minideam_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/minideam/agg/minideam_agg_features_librosa.csv',
    'anno_csv': '../data/minideam/agg/minideam_agg_annotations.csv'
}

dixon = {
    'name': 'dixon',
    'extraction': 'librosa',
    'feat_csv': '../data/dixon/dixon_features_librosa.csv',
    'anno_csv': '../data/dixon/dixon_labels.csv'
}
dixon_agg = {
    'name': 'dixon_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/dixon/agg/dixon_agg_features_librosa.csv',
    'anno_csv': '../data/dixon/agg/dixon_labels.csv'
}

bal_dixon = {
    'name': 'bal_dixon',
    'extraction': 'librosa',
    'feat_csv': '../data/dixon/bal_dixon_features_librosa.csv',
    'anno_csv': '../data/dixon/bal_dixon_labels.csv'
}

pmdeamo = {
    'name': 'pmdeamo',
    'extraction': 'librosa',
    'feat_csv': '../data/pmdeamo/pmdeamo_features_librosa.csv',
    'anno_csv': '../data/pmdeamo/pmdeamo_annotations.csv'
}
pmdeamo_agg = {
    'name': 'pmdeamo_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/pmdeamo/agg/pmdeamo_agg_features_librosa.csv',
    'anno_csv': '../data/pmdeamo/agg/pmdeamo_agg_annotations.csv'
}

bal_pmdeamo = {
    'name': 'bal_pmdeamo',
    'extraction': 'librosa',
    'feat_csv': '../data/pmdeamo/bal_pmdeamo_features_librosa.csv',
    'anno_csv': '../data/pmdeamo/bal_pmdeamo_annotations.csv'
}
bal_pmdeamo_agg = {
    'name': 'bal_pmdeamo_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/pmdeamo/agg/bal_pmdeamo_agg_features_librosa.csv',
    'anno_csv': '../data/pmdeamo/agg/bal_pmdeamo_agg_annotations.csv'
}

dimensions_list = ['valence', 'arousal']
ground_truth_cols_dict = {'valence': ['song_id', 'valence'],
                          'arousal': ['song_id', 'arousal'],
                          'label': ['song_id', 'label']}

DATASETS = {
    'pmemo_o': pmemo_o,
    'pmemo': pmemo,
    'pmemo_agg': pmemo_agg,
    'bal_pmemo': bal_pmemo,
    'bal_pmemo_agg': bal_pmemo_agg,
    'deam_o': deam_o,
    'deam': deam,
    'deam_agg': deam_agg,
    'bal_deam': bal_deam,
    'bal_deam_agg': bal_deam_agg,
    'minideam': minideam,
    'minideam_agg': minideam_agg,
    'dixon': dixon,
    'bal_dixon': bal_dixon,
    'dixon_agg': dixon_agg,
    'pmdeamo': pmdeamo,
    'pmdeamo_agg': pmdeamo_agg,
    'bal_pmdeamo': bal_pmdeamo,
    'bal_pmdeamo_agg': bal_pmdeamo_agg
}

# overall variables to edit when testing/experimenting
write_to_file = False
write_predictions = False
results_dir = '../data/results/'
date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

print('*** REPLICATION ***')
# replication parameters based on original pmemo researchers factors
initial_scaling = 'standard'
initial_n_comp = None
initial_w = False

# replication used for code verification
# replication_main = replication_experiment(['pmemo_o'])

# feature extraction comparison (openSMILE and libROSA) with same settings as replication experiment
# feature_ext_comparison = replication_experiment(['pmemo', 'deam_o', 'deam'])

print('*** SCALING AND REDUCTION ***')
# scale_reduce_main_c = scaling_and_reduction_comp_experiment(['pmemo', 'deam', 'dixon'])
# scale_reduce_bal_c = scaling_and_reduction_comp_experiment(['bal_pmemo', 'bal_deam'])
# scale_and_reduce_big_c = scaling_and_reduction_comp_experiment(['pmdeamo', 'bal_pmdeamo'])

print('*** NARROWING DOWN STEP PARAMETERS ***')
# based on experiments with polynomial kernel and degree, we fix degree to 1 for the remainder of experiments
# REGS['SVRpol'] = SVR(kernel='poly', gamma='scale', degree=1)
# CLFS['SVMpol'] = SVC(kernel='poly', gamma='scale', degree=1, decision_function_shape='ovo')

# excluding minmax-scaling on account of high sensitivity to outliers
# scales = ['standard', 'robust']

# choosing some select values for n-components
# comps = [None, 1, 2, 3, 4, 5, 10, 25, 50]
# chosen_w = True

# chosen_regs = list(REGS.keys())
# for scale in scales:
#     for comp in comps:
#         chosen_scaling = scale
#         chosen_n_comp = comp
#         baseline_main = baseline_regression(['pmemo', 'deam'])

# ************************************************************************************

print('*** BASELINE & TRANSFER | DIMENSIONAL TO DIMENSIONAL ***')
chosen_scaling = 'standard'
chosen_n_comp = 25
chosen_w = True
chosen_regs = ['Ridge', 'SVRrbf', 'SVRlin']

print('\nMAIN DATASETS ')
main_baseline = baseline_regression(['pmemo', 'deam'])
pmemo_to_deam = dim_to_dim_transfer_experiment(pmemo, deam)
deam_to_pmemo = dim_to_dim_transfer_experiment(deam, pmemo)

print('\nEQUAL-SIZED DATASETS')
minideam_baseline = baseline_regression(['minideam'])
pmemo_to_minideam = dim_to_dim_transfer_experiment(pmemo, minideam)
minideam_to_pmemo = dim_to_dim_transfer_experiment(minideam, pmemo)

print('\n*** PREDICT VA VALUES ON CATEGORICAL DATASET ***')
# predict VA values on categorical dataset by fitting scaler, pca and regressor to a dimensional dataset
print('\nMAIN DATASETS')
cat_baseline = baseline_classification(['dixon'])
pmemo_to_dixon = dim_to_cat_transfer_experiment(pmemo, dixon)
deam_to_dixon = dim_to_cat_transfer_experiment(deam, dixon)

print('\nEMOTION-BALANCED DATASETS')
bal_baseline = baseline_regression(['bal_pmemo', 'bal_deam'])
bal_pmemo_to_dixon = dim_to_cat_transfer_experiment(bal_pmemo, dixon)
bal_deam_to_dixon = dim_to_cat_transfer_experiment(bal_deam, dixon)

print('\nBIG DATASET')
# pmemo and deam combined into one bigger dataset
pmdeamo_baseline = baseline_regression(['pmdeamo'])
pmdeamo_to_dixon = dim_to_cat_transfer_experiment(pmdeamo, dixon)

# balanced big dataset
pmdeamo_bal_baseline = baseline_regression(['bal_pmdeamo'])
bal_pmdeamo_to_dixon = dim_to_cat_transfer_experiment(bal_pmdeamo, dixon)
