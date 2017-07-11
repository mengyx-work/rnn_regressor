import os, collections

# the namedtuple for dataset
train_data = collections.namedtuple('train_data', ['time_series_data',
                                                   'meta_data',
                                                   'target'])
# the namedtuple for column names
data_columns = collections.namedtuple('data_columns', ['time_step_list',
                                                       'time_interval_columns',
                                                       'static_columns',
                                                       'target_column'])


def check_expected_config_keys(local_config_dict, expected_keys):
    for key in expected_keys:
        if key not in local_config_dict:
            raise ValueError('failed to find necessary key {} in config_dict...'.format(key))


def create_column_config(config_dict):
    """function to load the configuration yaml file, create and
    return a `data_columns` named tuple.

    Args:
        config_dict (Dict) : the dict from configuration yaml file

    Returns:
        a `data_columns` named tuple.
    """

    expected_keys = ['index_column',
                     'label_column',
                     'static_columns',
                     'time_interval_columns',
                     'time_step_list']
    check_expected_config_keys(config_dict, expected_keys)
    columns = data_columns(time_step_list=config_dict['time_step_list'],
                           time_interval_columns=config_dict['time_interval_columns'],
                           static_columns=config_dict['static_columns'],
                           target_column=config_dict['label_column'])
    return columns


def full_column_name_by_time(col_prefix, time_stamp_appendix):
    return "{}_{}".format(col_prefix, time_stamp_appendix)


def all_expected_data_columns(config_dict):
    expected_columns = config_dict['static_columns'] + [config_dict['label_column']]
    for time_stamp in config_dict['time_step_list']:
        for name in config_dict['time_interval_columns']:
            expected_columns.append(full_column_name_by_time(name, time_stamp))
    return expected_columns


def clear_folder(absolute_folder_path):
    if not os.path.exists(absolute_folder_path):
        os.makedirs(absolute_folder_path)
        return
    for file_name in os.listdir(absolute_folder_path):
        file_path = os.path.join(absolute_folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print 'failed to clear folder {}, with error {}'.foramt(absolute_folder_path, e)
