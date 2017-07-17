import os
from series_data_generator import SeriesDataGenerator
from hybrid_model import HybridModel, model_predict, create_local_model_path
from utils import load_training_data_from_gcs
from data_preprocess import create_train_valid_data

GCS_path = 'test/ML'
yaml_file_name = 'training_configuration.yaml'
fraction = 0.7
model_name = "NYDN_hybrid_model"

config_dict, local_data_file = load_training_data_from_gcs(GCS_path, yaml_file_name)
train, valid_data = create_train_valid_data(local_data_file, config_dict.copy(), fraction)
data_generator = SeriesDataGenerator(train, config_dict)
test_generator = SeriesDataGenerator(valid_data, config_dict)
try:
    #tf.reset_default_graph()
    model = HybridModel(config_dict, model_name)
    model.train(data_generator, test_generator)
except Exception, e:
    print "found the exception {} in model training.".format(e)
finally:
    os.unlink(local_data_file)

pred_op_name = "NYDN_hybrid_model/fully_connect_layer/fully_connect_layer_2/Relu:0"
local_model_path = create_local_model_path(HybridModel.COMMON_PATH, model_name)
combined_results = model_predict(valid_data, config_dict.copy(), local_model_path, pred_op_name)
combined_results.to_csv("../combined_pred_results.csv", header=True)