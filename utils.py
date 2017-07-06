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

'''
## a namedtuple of feature names
independent_features = ['articleInfo_type', 'articleInfo_authorName', 'articleInfo_section', 'minLocalDateInWeek', 'minLocalTime', 'createTime', 'publishTime']
feature_name_strings = ['views_PageView', 'sessionReferrer_DIRECT_PageView', 'pageReferrer_OTHER_PageView', 'platform_PHON_PageView', 'platform_DESK_PageView', 'sessionReferrer_SEARCH_PageView', 'pageReferrer_SEARCH_PageView', 'platform_TBLT_PageView', 'pageReferrer_DIRECT_PageView', 'sessionReferrer_SOCIAL_PageView', 'pageReferrer_EMPTY_DOMAIN_PageView', 'pageReferrer_SOCIAL_PageView']
step_time_strings = ['0min_to_10min', '10min_to_20min', '20min_to_30min', '30min_to_40min', '40min_to_50min', '50min_to_60min']

columns = data_columns(step_time_strings=step_time_strings,
                       feature_name_strings=feature_name_strings,
                       meta_data_columns=independent_features,
                       target_column=label_col)
'''


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
