import os, yaml, collections

EXPECTED_CONFIG_KEYS = ['index_column',
                        'label_column',
                        'static_columns',
                        'time_interval_columns',
                        'time_step_list']

CONFIG_FILE_NAME = "training_configuration.yaml"

# the namedtuple for dataset
train_data = collections.namedtuple('train_data', ['time_series_data',
                                                   'meta_data',
                                                   'target'])
# the namedtuple for column names
data_columns = collections.namedtuple('data_columns', ['time_step_list',
                                                       'time_interval_columns',
                                                       'static_columns',
                                                       'target_column'])


def create_column_config(yaml_file_path=None):
    """function to load the configuration yaml file, create and
    return a `data_columns` named tuple.

    Args:
        yaml_file_path (String) : the file path for a yaml file

    Returns:
        a `data_columns` named tuple.
    """

    if yaml_file_path is None:
        yaml_file_path = os.path.join("./", CONFIG_FILE_NAME)

    if not os.path.exists(yaml_file_path):
        raise ValueError("failed to locate the configuration yaml file {}...".format(yaml_file_path))

    with open(yaml_file_path, 'r') as yaml_file:
        config_dict = yaml.load(yaml_file)

    for config_key in EXPECTED_CONFIG_KEYS:
        if config_key not in config_dict:
            raise ValueError("the expected key {} is not found in the configuration yaml file {}....".format(config_key, yaml_file_path))

    columns = data_columns(time_step_list=config_dict['time_step_list'],
                           time_interval_columns=config_dict['time_interval_columns'],
                           static_columns=config_dict['static_columns'],
                           target_column=config_dict['label_column'])
    return columns


def full_column_name_by_time(col_prefix, time_stamp_appendix):
    return "{}_{}".format(col_prefix, time_stamp_appendix)


def all_expected_data_columns():
    columns = create_column_config()
    expected_columns = columns.static_columns + [columns.target_column]
    for time_stamp in columns.time_step_list:
        for name in columns.time_interval_columns:
            expected_columns.append(full_column_name_by_time(name, time_stamp))
    return expected_columns


def clear_folder(absolute_folder_path):
    if not os.path.exists(absolute_folder_path):
        os.makedirs(absolute_folder_path)
    for file_name in os.listdir(absolute_folder_path):
        file_path = os.path.join(absolute_folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print 'failed to clear folder {}, with error {}'.foramt(absolute_folder_path, e)
