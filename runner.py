import os
from series_data_generator import SeriesDataGenerator
from hybrid_model import hybrid_model
from utils import load_data_from_gcs
from data_preprocess import create_train_valid_data

GCS_path = 'test/ML'
yaml_file_name = 'training_configuration.yaml'
fraction = 0.7

config_dict, local_data_file = load_data_from_gcs(GCS_path, yaml_file_name)
train, valid_data = create_train_valid_data(local_data_file, config_dict.copy(), fraction)

data_generator = SeriesDataGenerator(train, config_dict)
test_generator = SeriesDataGenerator(valid_data, config_dict)

try:
    #tf.reset_default_graph()
    model = hybrid_model(config_dict, "NYDN_hybrid_model")
    model.train(data_generator, test_generator)
except Exception, e:
    print "found the exception {} in model training.".format(e)
finally:
    os.unlink(local_data_file)

