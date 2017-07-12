import os
import pandas as pd
from random import shuffle
from utils import all_expected_data_columns, check_expected_config_keys


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

