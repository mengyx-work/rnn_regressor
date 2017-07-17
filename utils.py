import os, collections
import tempfile, yaml
from google_cloud_storage_util import GCS_Bucket

# the namedtuple for dataset
train_data = collections.namedtuple('train_data', ['time_series_data',
                                                   'meta_data',
                                                   'target'])


def model_meta_file(model_path, file_prefix="final_model"):
    meta_files = [f for f in os.listdir(model_path) if f[-5:] == '.meta']
    final_model_files = [f for f in meta_files if file_prefix in f]
    if len(final_model_files) == 0:
        raise ValueError("failed to find any model meta files in {}".format(model_path))
    if len(final_model_files) > 1:
        print "warning, more than one model meta file is found in {}".format(model_path)
    return os.path.join(model_path, final_model_files[0])


def process_target_list(nested_list):
    return [int(elem[0]) for elem in nested_list]


def check_expected_config_keys(local_config_dict, expected_keys):
    for key in expected_keys:
        if key not in local_config_dict:
            raise ValueError('failed to find necessary key {} in config_dict...'.format(key))


def load_training_data_from_gcs(GCS_path, yaml_file_name):
    local_data_file = tempfile.NamedTemporaryFile(delete=True).name
    expected_keys = ["time_interval_columns",
                     "static_columns",
                     "time_step_list",
                     "GCS_path",
                     "data_file_name",
                     "label_column",
                     "index_column"]
    bucket = GCS_Bucket("newsroom-backend")
    with tempfile.NamedTemporaryFile(delete=True) as yaml_file:
        bucket.take("{}/{}".format(GCS_path, yaml_file_name), yaml_file.name)
        config_dict = yaml.load(yaml_file)
        print "local yaml file: {}".format(yaml_file.name)

    check_expected_config_keys(config_dict, expected_keys)
    bucket.take("{}/{}".format(config_dict['GCS_path'], config_dict['data_file_name']), local_data_file)
    print "local data file: {}".format(local_data_file)
    return config_dict, local_data_file


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
