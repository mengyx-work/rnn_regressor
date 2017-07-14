import os
from series_data_generator import SeriesDataGenerator
from hybrid_model import hybrid_model, model_predict
from utils import load_training_data_from_gcs
from data_preprocess import create_train_valid_data

GCS_path = 'test/ML'
yaml_file_name = 'training_configuration.yaml'
fraction = 0.7
model_name = "NYDN_hybrid_model"

config_dict, local_data_file = load_training_data_from_gcs(GCS_path, yaml_file_name)
train, valid_data = create_train_valid_data(local_data_file, config_dict.copy(), fraction)
local_model_path = hybrid_model(config_dict, model_name).get_model_path()
pred_op_name = "NYDN_hybrid_model/fully_connect_layer/fully_connect_layer_2/Relu:0"
combined_results = model_predict(valid_data, config_dict.copy(), local_model_path, pred_op_name)
combined_results.to_csv("../combined_pred_results.csv", header=True)
