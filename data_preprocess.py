import os, yaml, tempfile
import pandas as pd
from random import shuffle
from utils import all_expected_data_columns, check_expected_config_keys, GCS_BUCKET_NAME
from google_cloud_storage_util import GCS_Bucket


def create_kfold_data_index_yaml_files(GCS_path, yaml_file_name, yaml_GCS_path, yaml_file_prefix, fold_num):
    '''function creates multiple idnex yaml files and store them
    in GCS.

    1. load the config_dict using `GCS_path` and `yaml_file_name`.
    2. load the full training data using config_dict
    3. split the data index into `fold_num` buckets and save
    `train_index` and `test_index` into yaml files
    4. store these yaml files to GCS

    args:
        GCS_path (string): the GCS path for model configuration file
        yaml_file_name (string): yaml configuration file name in GCS
        yaml_GCS_path (string): the GCS path to store index yaml file
        yaml_file_prefix: prefix for index yaml file in GCS
        fold_num (int): the number of folds to split data
    returns:
        yaml_file_list (list): a list of yaml files stored in GCS
        model_name_list (list): a list of model names
    '''
    config_dict, local_data_file = load_training_data_from_gcs(GCS_path, yaml_file_name)
    index_data = pd.read_csv(local_data_file,
                             index_col=config_dict["index_column"],
                             usecols=[config_dict["index_column"]])
    index = index_data.index.tolist()
    shuffle(index)
    bucket = GCS_Bucket(GCS_BUCKET_NAME)
    yaml_file_list = []
    model_name_list = []
    tot_length = len(index)
    chunk_size = int(tot_length / fold_num)
    for i in range(fold_num):
        start_index = i * chunk_size
        if i == (fold_num - 1):
            end_index = tot_length
        else:
            end_index = (i + 1) * chunk_size
        index_dict = {"test_index": index[start_index:end_index],
                      "train_index": index[0:start_index] + index[end_index:tot_length]}
        local_yaml_file = tempfile.NamedTemporaryFile(delete=True).name
        model_name = "{}_fold_{}".format(yaml_file_prefix, i + 1)
        yaml_file_name = "{}.yaml".format(model_name)
        model_name_list.append(model_name)
        yaml_file_list.append(yaml_file_name)
        print local_yaml_file, yaml_file_name
        with open(local_yaml_file, 'w') as yaml_file:
            yaml.dump(index_dict, yaml_file)
        bucket.put(local_yaml_file, "{}/{}".format(yaml_GCS_path, yaml_file_name))
        os.unlink(local_yaml_file)  # remove the local index yaml file
    os.unlink(local_data_file)  # remove the local data
    print yaml_file_list, model_name_list
    return yaml_file_list, model_name_list


def load_training_data_from_gcs(GCS_path, yaml_file_name):
    local_data_file = tempfile.NamedTemporaryFile(delete=True).name
    expected_keys = ["time_interval_columns",
                     "static_columns",
                     "time_step_list",
                     "GCS_path",
                     "label_column",
                     "index_column",
                     "data_file_name"]
    bucket = GCS_Bucket("newsroom-backend")
    config_dict = load_yaml_file_from_gcs(bucket, GCS_path, yaml_file_name)
    check_expected_config_keys(config_dict, expected_keys)
    bucket.take("{}/{}".format(config_dict['GCS_path'], config_dict['data_file_name']), local_data_file)
    print "local data file: {}".format(local_data_file)
    return config_dict, local_data_file


def load_yaml_file_from_gcs(bucket, GCS_path, yaml_file_name):
    with tempfile.NamedTemporaryFile(delete=True) as yaml_file:
        bucket.take("{}/{}".format(GCS_path, yaml_file_name), yaml_file.name)
        config_dict = yaml.load(yaml_file)
        print "local yaml file: {}".format(yaml_file.name)
    return config_dict


def create_train_test_by_index(local_data_file, config_dict, index_dict):
    data = pd.read_csv(local_data_file, index_col=config_dict["index_column"])
    for column in all_expected_data_columns(config_dict):
        if column not in data.columns:
            raise ValueError("failed to find column {} in data {}".format(column, local_data_file))
    valid_data = data.loc[index_dict['test_index']]
    train = data.loc[index_dict['train_index']]
    print "total #columns: {}, training data: {}, validation data: {}.".format(train.shape[1],
                                                                               train.shape[0],
                                                                               valid_data.shape[0])
    return train, valid_data


def create_train_valid_data(data_path, config_dict, frac=0.6):
    """
    Args:
        data_path (String) : the data path
        config_dict (Dict) : the config_dict from top runner
        frac (double) : the fraction of data used as validation

    Returns:
        train (Pandas DataFrame) : the training data
        valid_data (Pandas DataFrame) : the validation data
    """
    if not os.path.exists(data_path):
        raise ValueError("failed to locate the data file {}.".format(data_path))
    data = pd.read_csv(data_path, index_col=config_dict["index_column"])
    for column in all_expected_data_columns(config_dict):
        if column not in data.columns:
            raise ValueError("failed to find column {} in data {}".format(column, data_path))
    train_index, valid_index = _create_validation_index(data, config_dict["label_column"], frac, to_shuffle=False, group_by_dep_var=False)
    valid_data = data.loc[valid_index]
    train = data.loc[train_index]
    print "split by fraction {}, training #rows: {}, validation #rows: {}".format(frac, train.shape[0], valid_data.shape[0])
    return train, valid_data


def _create_validation_index(df, dep_var_name, train_frac=0.8, to_shuffle=False, group_by_dep_var=False):
    """function to generate two sets of index: `valid_index` and `train_idex`

    Args:
        df (Pandas DataFrame) : the data to generate index from
        dep_var_name (String) : the index column name
        valid_frac (double) : the fraction of validation data
        to_shuffle (bool) : whether to shuffle the index
        group_by_dep_var (bool) : whether to group by dep_var when generating index

   Returns:
       train_index (List of index) : index for training data
       valid_index (List of index) : index for validation data
    """
    valid_index = []
    train_index = []

    if group_by_dep_var:
        index_series = df[dep_var_name]
        grouped_index = index_series.groupby(index_series)

        for name, group in grouped_index:
            index_length = int(1. * train_frac * group.shape[0])
            train_index.extend(group[0:index_length].index.tolist())
            valid_index.extend(group[index_length:].index.tolist())

    train_index = df.index.to_series().sample(int(df.shape[0] * 1. * train_frac), replace=False)
    valid_index = list(set(df.index.to_series()) - set(train_index))
    # shuffle the training and test data in place
    if to_shuffle:
        shuffle(train_index)
        shuffle(valid_index)
    print 'before sending ', len(train_index), len(valid_index)
    return train_index, valid_index

