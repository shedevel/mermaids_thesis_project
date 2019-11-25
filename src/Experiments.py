from datetime import datetime
import pandas as pd
from math import sqrt
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Lasso, ElasticNet, Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
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
    return split_into_features_and_targets(dataset)


# combines features and annotations by appending the target column, returns pandas dataframe
def combine_features_and_annotations(features_csv, annotation_csv, id_and_dim_cols):
    features = pd.read_csv(features_csv, index_col=0)
    annotations = pd.read_csv(annotation_csv, index_col=0, usecols=id_and_dim_cols)

    data_df = features.join(annotations)
    return data_df.dropna()


# assumes that the targets are the last column, returns two series array-like objects
def split_into_features_and_targets(data):
    features = data[data.columns[:-1]].values
    targets = data[data.columns[-1]].values
    return features, targets


# rmse is not a part of the sklearn metrics
def rmse(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))


# 10-fold cross validation with supplied parameters
def cross_val_regression_rmse(features_list, y_true_list, scaler, regressor):
    scorer = {'rmse': make_scorer(rmse)}

    reg = make_pipeline(scaler, PCA(exp_var_ratio, whiten=w), regressor)
    scores_dict = cross_validate(reg, features_list, y_true_list, scoring=scorer, cv=10, return_train_score=False)
    mean_rmse_score = (scores_dict['test_rmse'].mean())
    return mean_rmse_score


# 10-fold cross validation with supplied parameters
def cross_val_classification(features, y_true, scaler, classifier):
    clf = make_pipeline(scaler, PCA(exp_var_ratio, whiten=w), classifier)
    scores = cross_validate(clf, features, y_true, cv=10, return_train_score=False)
    mean_acc_score = (scores['test_score'].mean())
    return mean_acc_score


# used for verification of regression code
def replication_experiment(dataset_names):
    delim = ','

    replication = list()
    for name in dataset_names:
        data = DATASETS[name]

        for dim in dimensions_list:
            features, targets = prepare_dataset(data['feat_csv'], data['anno_csv'], ground_truth_cols_dict[dim])

            for reg in REGS:
                mean_rmse = cross_val_regression_rmse(features, targets, SCALERS[scaling_str], REGS[reg])
                row_string = dim + delim + name + delim + data['extraction'] + delim + scaling_str + delim + str(
                    exp_var_ratio) + delim + str(w) + delim + reg + delim + 'rmse' + delim + '{:.3f}'.format(
                    mean_rmse) + delim + date_str
                print(row_string)
                replication.append(row_string + '\n')

                if write_to_file:
                    with open(results_dir + 'replication_and_feature_extraction_comparison.csv', 'a') as output:
                        output.write(row_string + '\n')
    return replication


# investigating scaling and pca parameters for "optimal" combination
def scaling_and_reduction_experiment(dataset_names):
    delim = ','

    scaling_and_reduction = list()
    for name in dataset_names:
        data = DATASETS[name]
        features, targets = prepare_dataset(data['feat_csv'], data['anno_csv'], [0, 1])

        for scaler_name in SCALERS:
            scaler = SCALERS[scaler_name]
            features = scaler.fit_transform(features)

            pca_val = exp_var_ratio_min
            while pca_val <= exp_var_ratio_max:
                pca = PCA(pca_val)
                pca.fit(features)

                row_string = name + delim + data['extraction'] + delim + scaler_name + delim + '{:.2f}'.format(
                    pca_val) + delim + str(pca.n_components_) + delim + date_str
                print(row_string)
                scaling_and_reduction.append(row_string + '\n')

                if write_to_file:
                    with open(results_dir + 'scaling_and_reduction.csv', 'a') as output:
                        output.write(row_string + '\n')

                pca_val += .10
    return scaling_and_reduction


# baseline regression on dimensional datasets before transfer
def baseline_regression(dataset_names):
    delim = ','

    baseline = list()
    for name in dataset_names:
        data = DATASETS[name]

        for dim in dimensions_list:
            features, targets = prepare_dataset(data['feat_csv'], data['anno_csv'], ground_truth_cols_dict[dim])

            for reg_name in list_of_regs:
                mean_rmse = cross_val_regression_rmse(features, targets, SCALERS[scaling_str], REGS[reg_name])
                row_string = dim + delim + name + delim + data['extraction'] + delim + scaling_str + delim + str(
                    exp_var_ratio) + delim + str(w) + delim + reg_name + delim + 'rmse' + delim + '{:.3f}'.format(
                    mean_rmse) + delim + date_str
                print(row_string)
                baseline.append(row_string + '\n')

                if write_to_file:
                    with open(results_dir + 'baseline.csv', 'a') as output:
                        output.write(row_string + '\n')
    return baseline


# baseline classification on categorical dataset before transfer
def baseline_classification(dataset_names):
    delim = ','

    baseline = list()
    for name in dataset_names:
        data = DATASETS[name]
        features, targets = prepare_dataset(data['feat_csv'], data['anno_csv'], ground_truth_cols_dict['label'])

        for clf in CLFS:
            mean_acc = cross_val_classification(features, targets, SCALERS[scaling_str], CLFS[clf])
            row_string = 'categorical' + delim + name + delim + data['extraction'] + delim + scaling_str + delim + str(
                exp_var_ratio) + delim + str(w) + delim + clf + delim + 'accuracy' + delim + '{:.2f}'.format(
                mean_acc) + delim + date_str
            print(row_string)
            baseline.append(row_string + '\n')

            if write_to_file:
                with open(results_dir + 'baseline.csv', 'a') as output:
                    output.write(row_string + '\n')
    return baseline


# preparing the datasets for transfer, first combine then split, returns dict
def prepare_dimensional_dataset_for_transfer(data_dict):
    data_df = combine_dataset(data_dict['feat_csv'], data_dict['anno_csv'])
    feat_nda, valence_nda, arousal_nda = split_features_and_dim_annotations(data_df)
    rows_list, cols_list = get_row_and_column_headers(data_df)

    return {'name': data_dict['name'], 'features': feat_nda, 'valence': valence_nda, 'arousal': arousal_nda,
            'cols': cols_list, 'rows': rows_list}


# preparing the datasets for transfer, first combine then split, returns dict
def prepare_categorical_dataset_for_transfer(data_dict):
    data_df = combine_dataset(data_dict['feat_csv'], data_dict['anno_csv'])
    feat_nda, lab_nda = split_features_and_cat_annotations(data_df)
    rows_list, cols_list = get_row_and_column_headers(data_df)

    return {'name': data_dict['name'], 'features': feat_nda, 'labels': lab_nda,
            'cols': cols_list, 'rows': rows_list}


# join features and annotations into one dataframe
def combine_dataset(features_csv_str, annotations_csv_str):
    features_df = pd.read_csv(features_csv_str, index_col='song_id')
    annotations_df = pd.read_csv(annotations_csv_str, index_col='song_id')
    data_df = features_df.join(annotations_df)
    return data_df.dropna()


# split dimensional dataset into three 2d-arrays: features, valence and arousal
def split_features_and_dim_annotations(dataset_df):
    valence_nda = dataset_df['valence'].values
    arousal_nda = dataset_df['arousal'].values
    features_nda = dataset_df[[c for c in dataset_df.columns if c not in ['valence', 'arousal', 'label']]].values
    return features_nda, valence_nda, arousal_nda


# split categorical dataset into two 2d-arrays: features and labels
def split_features_and_cat_annotations(dataset_df):
    labels_nda = dataset_df['label'].values
    features_nda = dataset_df.drop(columns=['label']).values
    return features_nda, labels_nda


# takes a dataframe and returns two lists
def get_row_and_column_headers(dataset_df):
    rows = dataset_df.index.values.tolist()
    cols = dataset_df[[c for c in dataset_df.columns if c not in ['valence', 'arousal', 'label']]].columns.tolist()
    return rows, cols


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


# fit scaling, pca and regressor to first dataset, then predict VA-values on second dataset
def dim_to_dim_transfer_experiment(from_data, to_data):
    train = prepare_dimensional_dataset_for_transfer(from_data)
    test = prepare_dimensional_dataset_for_transfer(to_data)

    scaler = SCALERS[scaling_str]

    # fitting the scaler to the train data features, then transforming test and train features
    scaler.fit(train['features'])
    train['features'] = scaler.transform(train['features'])
    test['features'] = scaler.transform(test['features'])

    # fitting the pca to the training features
    pca = PCA(exp_var_ratio, whiten=w)
    pca.fit(train['features'])

    # transforming both train and test features with the fitted pca
    train_pc = pca.transform(train['features'])
    test_pc = pca.transform(test['features'])

    delim = ','
    transfer = list()
    for reg_name in list_of_regs:
        ground_truth_and_pred_vals_df = pd.DataFrame(index=test['rows'])
        ground_truth_and_pred_vals_df['dataset'] = test['name']
        ground_truth_and_pred_vals_df['train_set'] = train['name']
        ground_truth_and_pred_vals_df['scaling'] = scaling_str
        ground_truth_and_pred_vals_df['pca_val'] = exp_var_ratio

        regressor = REGS[reg_name]
        for dim in dimensions_list:
            # training the regressor on train principal components, testing on test principal components
            regressor.fit(train_pc, train[dim])
            pred_vals_list = regressor.predict(test_pc)

            rmse_score = rmse(test[dim], pred_vals_list)
            rmse_str = '{:.3f}'
            row_string = dim + delim + train['name'] + delim + test['name'] + delim + scaling_str + delim + str(
                exp_var_ratio) + delim + str(w) + delim + reg_name + delim + 'rmse' + delim + rmse_str.format(
                rmse_score) + delim + date_str
            print(row_string)
            transfer.append(row_string)

            if write_to_file:
                with open(results_dir + 'transfer.csv', 'a') as output:
                    output.write(row_string + '\n')

            # combine ground truth and predicted vals for writing results to csv
            truth_and_pred_vals_df = combine_annotations_and_predicted_vals(dim, test['rows'], test[dim],
                                                                            pred_vals_list)
            ground_truth_and_pred_vals_df = ground_truth_and_pred_vals_df.join(truth_and_pred_vals_df)

        if write_to_file:
            ground_truth_and_pred_vals_df.to_csv(
                results_dir + train['name'] + '_' + test['name'] + '_' + scaling_str + '_' + str(
                    exp_var_ratio) + '_' + str(w) + '_' + reg_name + '_' + date_str + '.csv',
                index_label='song_id')
    return transfer


# fit scaling, pca and regressor to first dataset, then predict VA-values on second dataset
def dim_to_cat_transfer_experiment(dim_data, cat_data):
    train = prepare_dimensional_dataset_for_transfer(dim_data)
    test = prepare_categorical_dataset_for_transfer(cat_data)

    scaler = SCALERS[scaling_str]

    # fitting the scaler to the train data features
    scaler.fit(train['features'])
    train['features'] = scaler.transform(train['features'])
    test['features'] = scaler.transform(test['features'])

    # fitting the pca to the training features
    pca = PCA(exp_var_ratio, whiten=w)
    pca.fit(train['features'])

    # transforming both train and test features with the fitted pca
    train_pc = pca.transform(train['features'])
    test_pc = pca.transform(test['features'])

    transfer = list()
    pred_vals = dict()

    delim = ','
    for reg_name in list_of_regs:
        regressor = REGS[reg_name]
        for dim in dimensions_list:
            # training the regressor on train principal components, testing on test principal components
            regressor.fit(train_pc, train[dim])
            # adding predicted values for current dimension to dict
            pred_vals[dim] = regressor.predict(test_pc)

        truth_and_prediction_df = pd.DataFrame(index=test['rows'])
        truth_and_prediction_df['dataset'] = test['name']
        truth_and_prediction_df['train_set'] = train['name']
        truth_and_prediction_df['scaling'] = scaling_str
        truth_and_prediction_df['pca_val'] = exp_var_ratio
        truth_and_prediction_df = truth_and_prediction_df.join(
            combine_labels_and_VA_predictions(test['rows'], test['labels'], pred_vals['valence'], pred_vals['arousal']))

        # calculate accuracy based on true labels and predicted VA quadrant
        accuracy = accuracy_score(test['labels'], truth_and_prediction_df['pred_quadrant'])

        row_string = 'categorical' + delim + train['name'] + delim + test['name'] + delim + scaling_str + delim + str(
            exp_var_ratio) + delim + str(w) + delim + reg_name + delim + 'accuracy' + delim + '{:.2f}'.format(
            accuracy) + delim + date_str
        print(row_string)
        transfer.append(row_string + '\n')

        if write_to_file:
            with open(results_dir + 'transfer.csv', 'a') as output:
                output.write(row_string + '\n')
            truth_and_prediction_df.to_csv(
                results_dir + train['name'] + '_' + test['name'] + '_' + scaling_str + '_' + str(
                    exp_var_ratio) + '_' + str(w) + '_' + reg_name + '_' + date_str + '.csv',
                index_label='song_id')
    return transfer


# returns dataframe for writing results to csv for further investigation
def combine_labels_and_VA_predictions(rows, labels, pred_valence, pred_arousal):
    labels_series = pd.Series(labels, index=rows, name='labels')
    valence_series = pd.Series(pred_valence, index=rows, name='pred_valence')
    arousal_series = pd.Series(pred_arousal, index=rows, name='pred_arousal')
    combined_df = labels_series.to_frame().join(valence_series.to_frame().join(arousal_series.to_frame()))

    combined_df['pred_quadrant'] = [predicted_VA_quadrant(v, a) for v, a in
                                    zip(combined_df['pred_valence'], combined_df['pred_arousal'])]
    return combined_df


# based on the predicted VA-values, assign a predicted quadrant label such that accuracy can be calculated
def predicted_VA_quadrant(pred_valence, pred_arousal):
    if pred_valence <= 0.5:
        if pred_arousal > 0.5:
            return 'angry'
        else:
            return 'sad'
    else:
        if pred_arousal >= 0.5:
            return 'happy'
    return 'relax'


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
    'RandFor': RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1)
}

CLFS = {
    'Uniform': DummyClassifier(strategy='uniform'),
    'Ridge': RidgeClassifier(),
    'SVCrbf': SVC(kernel='rbf', gamma='scale', decision_function_shape='ovo'),
    'SVCpoly': SVC(kernel='poly', gamma='scale', decision_function_shape='ovo')
}

SCALERS = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler()
}

pmemo_o = {
    'name': 'pmemo',
    'extraction': 'opensmile',
    'feat_csv': '../data/pmemo/pmemo_static_features_opensmile.csv',
    'anno_csv': '../data/pmemo/pmemo_static_annotations.csv'
}
pmemo = {
    'name': 'pmemo',
    'extraction': 'librosa',
    'feat_csv': '../data/pmemo/pmemo_static_features_librosa.csv',
    'anno_csv': '../data/pmemo/pmemo_static_annotations.csv'
}
pmemo_agg = {
    'name': 'pmemo_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/pmemo/pmemo_agg_static_features_librosa.csv',
    'anno_csv': '../data/pmemo/pmemo_static_annotations.csv'
}

pmemobal = {
    'name': 'pmemobal',
    'extraction': 'librosa',
    'feat_csv': '../data/pmemo/pmemobal_static_features_librosa.csv',
    'anno_csv': '../data/pmemo/pmemo_static_annotations.csv'
}
pmemobal_agg = {
    'name': 'pmemobal_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/pmemo/pmemobal_agg_static_features_librosa.csv',
    'anno_csv': '../data/pmemo/pmemobal_agg_static_annotations.csv'
}

deam_o = {
    'name': 'deam',
    'extraction': 'opensmile',
    'feat_csv': '../data/deam/deam_static_features_opensmile.csv',
    'anno_csv': '../data/deam/deam_static_annotations.csv'
}
deam = {
    'name': 'deam',
    'extraction': 'librosa',
    'feat_csv': '../data/deam/deam_static_features_librosa.csv',
    'anno_csv': '../data/deam/deam_static_annotations.csv'
}
deam_agg = {
    'name': 'deam_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/deam/deam_agg_static_features_librosa.csv',
    'anno_csv': '../data/deam/deam_static_annotations.csv'
}

deambal = {
    'name': 'deambal',
    'extraction': 'librosa',
    'feat_csv': '../data/deam/deambal_static_features_librosa.csv',
    'anno_csv': '../data/deam/deambal_static_annotations.csv'
}

deambal_agg = {
    'name': 'deam_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/deam/deambal_agg_static_features_librosa.csv',
    'anno_csv': '../data/deam/deambal_agg_static_annotations.csv'
}

minideam = {
    'name': 'minideam',
    'extraction': 'librosa',
    'feat_csv': '../data/minideam/minideam_static_features_librosa.csv',
    'anno_csv': '../data/minideam/minideam_static_annotations.csv'
}
minideam_agg = {
    'name': 'minideam_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/minideam/minideam_agg_static_features_librosa.csv',
    'anno_csv': '../data/minideam/minideam_static_annotations.csv'
}

dixon = {
    'name': 'dixon',
    'extraction': 'librosa',
    'feat_csv': '../data/dixon/dixon_static_features_librosa.csv',
    'anno_csv': '../data/dixon/dixon_static_labels.csv'
}
dixon_agg = {
    'name': 'dixon_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/dixon/dixon_agg_static_features_librosa.csv',
    'anno_csv': '../data/dixon/dixon_static_labels.csv'
}

pmdeamo = {
    'name': 'pmdeamo',
    'extraction': 'librosa',
    'feat_csv': '../data/pmdeamo/pmdeamo_static_features_librosa.csv',
    'anno_csv': '../data/pmdeamo/pmdeamo_static_annotations.csv'
}
pmdeamo_agg = {
    'name': 'pmdeamo_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/pmdeamo/pmdeamo_agg_static_features_librosa.csv',
    'anno_csv': '../data/pmdeamo/pmdeamo_agg_static_annotations.csv'
}

pmdeamobal = {
    'name': 'pmdeamobal',
    'extraction': 'librosa',
    'feat_csv': '../data/pmdeamo/pmdeamobal_static_features_librosa.csv',
    'anno_csv': '../data/pmdeamo/pmdeamobal_static_annotations.csv'
}
pmdeamobal_agg = {
    'name': 'pmdeamobal_agg',
    'extraction': 'librosa',
    'feat_csv': '../data/pmdeamo/pmdeamobal_agg_static_features_librosa.csv',
    'anno_csv': '../data/pmdeamo/pmdeamobal_agg_static_annotations.csv'
}

dimensions_list = ['valence', 'arousal']
ground_truth_cols_dict = {'valence': ['song_id', 'valence'],
                          'arousal': ['song_id', 'arousal'],
                          'label': ['song_id', 'label']}

DATASETS = {
    'pmemo_o': pmemo_o,
    'pmemo': pmemo,
    'pmemo_agg': pmemo_agg,
    'pmemobal': pmemobal,
    'pmemobal_agg': pmemobal_agg,
    'deam_o': deam_o,
    'deam': deam,
    'deam_agg': deam_agg,
    'deambal': deambal,
    'deambal_agg': deambal_agg,
    'minideam': minideam,
    'minideam_agg': minideam_agg,
    'dixon': dixon,
    'dixon_agg': dixon_agg,
    'pmdeamo': pmdeamo,
    'pmdeamo_agg': pmdeamo_agg,
    'pmdeamobal': pmdeamobal,
    'pmdeamobal_agg': pmdeamobal_agg
}

# overall variables to edit when testing/experimenting
write_to_file = True
results_dir = '../data/results/'
date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

print('*** REPLICATION ***')
# replication parameters based on original pmemo researchers factors
scaling_str = 'standard'
exp_var_ratio = None
w = False

# used for code verification and feature extraction comparison, and initial baseline
main_datasets = ['pmemo_o', 'pmemo', 'deam_o', 'deam']
replication_main = replication_experiment(main_datasets)

# agg_datasets = ['pmemo_agg', 'deam_agg']
# replication_agg = replication_experiment(agg_datasets)

print('*** SCALING AND REDUCTION ***')
# pca reduction parameters to cross-test with every type of scaling
exp_var_ratio_min = .25
exp_var_ratio_max = .95

main_datasets = ['pmemo', 'deam', 'dixon']
scale_reduce_main = scaling_and_reduction_experiment(main_datasets)

# agg_datasets = ['pmemo_agg', 'deam_agg', 'dixon_agg']
# scale_reduce_agg = scaling_and_reduction_experiment(agg_datasets)

bal_datasets = ['pmemobal', 'deambal']
scale_reduce_bal = scaling_and_reduction_experiment(bal_datasets)

# bal_agg_datasets = ['pmemobal_agg', 'deambal_agg']
# scale_reduce_bal_agg = scaling_and_reduction_experiment(bal_agg_datasets)

print('*** BASELINE ***')
# pre-transfer baseline with the decided fixed factors regarding regressors, scaling and pca parameters
list_of_regs = ['Ridge', 'SVRrbf', 'SVRpoly']
scaling_str = 'robust'
exp_var_ratio = .75
w = False

main_datasets = ['pmemo', 'deam']
baseline = baseline_regression(main_datasets)

# agg_datasets = ['pmemo_agg', 'deam_agg']
# baseline_agg = baseline_regression(agg_datasets)

bal_datasets = ['pmemobal', 'deambal']
baseline_bal = baseline_regression(bal_datasets)

# bal_agg_datasets = ['pmemobal_agg', 'deambal_agg']
# baseline_list = baseline_regression(bal_agg_datasets)

print('*** TRANSFER DIMENSIONAL TO DIMENSIONAL ***')
# transfer with main datasets (same to same as a baseline, will be overfitted)
pmemo_to_pmemo = dim_to_dim_transfer_experiment(pmemo, pmemo)
pmemo_to_deam = dim_to_dim_transfer_experiment(pmemo, deam)
deam_to_deam = dim_to_dim_transfer_experiment(deam, deam)
deam_to_pmemo = dim_to_dim_transfer_experiment(deam, pmemo)

# transfer with main datasets with agg features
# pmemo_to_pmemo_agg = dim_to_dim_transfer_experiment(pmemo_agg, pmemo_agg)
# pmemo_to_deam_agg = dim_to_dim_transfer_experiment(pmemo_agg, deam_agg)
# deam_to_deam_agg = dim_to_dim_transfer_experiment(deam_agg, deam_agg)
# deam_to_pmemo_agg = dim_to_dim_transfer_experiment(deam_agg, pmemo_agg)

# transfer with equal-sized datasets (pmemo and minideam)
minideam_to_minideam = dim_to_dim_transfer_experiment(minideam, minideam)
pmemo_to_minideam = dim_to_dim_transfer_experiment(pmemo, minideam)
minideam_to_pmemo = dim_to_dim_transfer_experiment(minideam, pmemo)

# transfer with equal-sized datasets with agg features
# minideam_to_minideam_agg = dim_to_dim_transfer_experiment(minideam_agg, minideam_agg)
# pmemo_to_minideam_agg = dim_to_dim_transfer_experiment(pmemo_agg, minideam_agg)
# minideam_to_pmemo_agg = dim_to_dim_transfer_experiment(minideam_agg, pmemo_agg)

# transfer with emotion-balanced datasets
pmemobal_to_pmemobal = dim_to_dim_transfer_experiment(pmemobal, pmemobal)
pmemobal_to_deam = dim_to_dim_transfer_experiment(pmemobal, deam)
deambal_to_deambal = dim_to_dim_transfer_experiment(deambal, deambal)
deambal_to_pmemo = dim_to_dim_transfer_experiment(deambal, pmemo)

# transfer with emotion-balanced datasets with agg features
# pmemobal_to_pmemobal_agg = dim_to_dim_transfer_experiment(pmemobal_agg, pmemobal_agg)
# pmemobal_to_deam_agg = dim_to_dim_transfer_experiment(pmemobal_agg, deam_agg)
# deambal_to_deambal_agg = dim_to_dim_transfer_experiment(deambal_agg, deambal_agg)
# deambal_to_pmemo_agg = dim_to_dim_transfer_experiment(deambal_agg, pmemo_agg)

print('*** TRANSFER DIMENSIONAL TO CATEGORICAL ***')
# predict VA values on categorical dataset by fitting scaler, pca and regressor to a dimensional dataset
# main datasets
cat_baseline = baseline_classification(['dixon'])
pmemo_to_dixon = dim_to_cat_transfer_experiment(pmemo, dixon)
deam_to_dixon = dim_to_cat_transfer_experiment(deam, dixon)

# agg features
# cat_agg_baseline = baseline_classification(['dixon_agg'])
# pmemo_to_dixon_agg = dim_to_cat_transfer_experiment(pmemo_agg, dixon_agg)
# deam_to_dixon_agg = dim_to_cat_transfer_experiment(deam_agg, dixon_agg)

# pmemo and deam combined into one bigger dataset
baseline_regression(['pmdeamo', 'pmdeamobal'])
pmdeamo_to_dixon = dim_to_cat_transfer_experiment(pmdeamo, dixon)

# agg features
# baseline_regression(['pmdeamo_agg', 'pmdeamobal_agg'])
# pmdeamo_to_dixon_agg = dim_to_cat_transfer_experiment(pmdeamo_agg, dixon_agg)

# balanced dimensional datasets
pmemobal_to_dixon = dim_to_cat_transfer_experiment(pmemobal, dixon)
deambal_to_dixon = dim_to_cat_transfer_experiment(deambal, dixon)
pmdeamobal_to_dixon = dim_to_cat_transfer_experiment(pmdeamobal, dixon)

# agg features
# pmemobal_to_dixon_agg = dim_to_cat_transfer_experiment(pmemobal_agg, dixon_agg)
# deambal_to_dixon_agg = dim_to_cat_transfer_experiment(deambal_agg, dixon_agg)
# pmdeamobal_to_dixon_agg = dim_to_cat_transfer_experiment(pmdeamobal_agg, dixon_agg)
