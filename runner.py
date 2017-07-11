import tempfile, yaml
from series_data_generator import SeriesDataGenerator
from hybrid_model import hybrid_model
from utils import create_column_config, check_expected_config_keys
from data_preprocess import create_train_valid_data
from google_cloud_storage_util import GCS_Bucket

GCS_path = 'test/ML'
yaml_file_name = 'training_configuration.yaml'
local_yaml_file = tempfile.NamedTemporaryFile(delete=True).name
local_data_file = tempfile.NamedTemporaryFile(delete=True).name
expected_keys = ["time_interval_columns",
                 "static_columns",
                 "time_step_list",
                 "GCS_path",
                 "data_file_name",
                 "label_column",
                 "index_column"]
bucket = GCS_Bucket("newsroom-backend")
bucket.take("{}/{}".format(GCS_path, yaml_file_name), local_yaml_file)
with open(local_yaml_file, 'r') as yaml_file:
    config_dict = yaml.load(yaml_file)
check_expected_config_keys(config_dict, expected_keys)
bucket.take("{}/{}".format(config_dict['GCS_path'], config_dict['data_file_name']), local_data_file)

fraction = 0.7
config = create_column_config(config_dict.copy())
train, valid_data = create_train_valid_data(local_data_file, config_dict.copy(), fraction)

data_generator = SeriesDataGenerator(train, config)
test_generator = SeriesDataGenerator(valid_data, config)

#tf.reset_default_graph()
model = hybrid_model(config_dict)
#model.train(data_generator)
model.train(data_generator, test_generator)

