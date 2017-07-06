import os
import pandas as pd
from random import shuffle
from utils import all_expected_data_columns


def create_train_valid_data(file_path, file_name, label_col, index_col, frac=0.6):
    """

    Args:
        file_path (String) : the directory path
        file_name (String) : data file name
        label_col (String) : the dep_var/target column name
        index_col (String) : the index column name
        frac (double) : the fraction of data used as validation

    Returns:
        train (Pandas DataFrame) : the training data
        valid_data (Pandas DataFrame) : the validation data
    """

    data_path = os.path.join(file_path, file_name)
    if not os.path.exists(data_path):
        raise ValueError("failed to locate the data file {}.".format(data_path))
    data = pd.read_csv(data_path, index_col=index_col)
    for column in all_expected_data_columns():
        if column not in data.columns:
            raise ValueError("failed to find column {} in data {}".format(column, data_path))
    train_index, valid_index = _create_validation_index(data, label_col, frac, to_shuffle=True, group_by_dep_var=False)
    valid_data = data.loc[valid_index]
    train = data.loc[train_index]
    print "split by fraction {}, training #rows: {}, validation #rows: {}".format(frac, train.shape[0], valid_data.shape[0])
    return train, valid_data


def _create_validation_index(df, dep_var_name, valid_frac=0.2, to_shuffle=False, group_by_dep_var=False):
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
            index_length = int(valid_frac * group.shape[0])
            valid_index.extend(group[0:index_length].index.tolist())
            train_index.extend(group[index_length:].index.tolist())

    train_index = df.index.to_series().sample(int(df.shape[0] * 1. * valid_frac), replace=False)
    valid_index = list(set(df.index.to_series()) - set(train_index))

    # shuffle the training and test data in place
    if to_shuffle:
        shuffle(train_index)
        shuffle(valid_index)

    return train_index, valid_index

