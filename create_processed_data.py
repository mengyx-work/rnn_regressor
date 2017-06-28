import os, time, sys
import numpy as np
import pandas as pd
from columns_processor import categorical_columns_processor

sys.path.append('/Users/matt.meng/dev/xgboost_hyperopt')
from src.validation_tools import create_validation_index

label_col = 'total_views'
index_col = 'articleId'
#dep_var_col = 'total_views'

data_path = '/Users/matt.meng/Google_Drive/Taboola/ML/'
data_file = 'combined_data_60min_exposure_10min_window_24hr_target.csv'
processed_train_file = 'processed_train_data.csv'
processed_test_file = 'processed_test_data.csv'
data = pd.read_csv(os.path.join(data_path, data_file))
data = data.set_index(index_col)

frac = 0.6
train_index, valid_index = create_validation_index(data, label_col, frac, to_shuffle=True, group_by_dep_var=False)

valid_data  = data.ix[valid_index]
train       = data.ix[train_index]
valid_label = valid_data[label_col]
print train.shape
print valid_data.shape

category_columns = []
for col, dtype in zip(train.columns, train.dtypes):
    if dtype == 'object':
        category_columns.append(col)

columns_processor = categorical_columns_processor(train, valid_data, label_col) 
for col in category_columns:
    train.ix[:, col], valid_data.ix[:, col] = columns_processor.encode_categorical_column(column=col)

train.to_csv(os.path.join(data_path, processed_train_file), index=False, header=True)
valid_data.to_csv(os.path.join(data_path, processed_test_file), index=False, header=True)

