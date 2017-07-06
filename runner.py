import os, time, sys
import pandas as pd
from series_data_generator import SeriesDataGenerator
from hybrid_model import hybrid_model
from utils import create_column_config

data_path = '/Users/matt.meng/taboola_data/taboola_process_data'
processed_train_file = 'NYDN_240min_fullWindow_60min_exposure_120seconds_interval_target_24hr_data.csv'
#processed_test_file = 'processed_test_data.csv'
train = pd.read_csv(os.path.join(data_path, processed_train_file))
#test = pd.read_csv(os.path.join(data_path, processed_test_file))

data_generator = SeriesDataGenerator(train, create_column_config())
#test_generator = SeriesDataGenerator(test, create_column_config())

#tf.reset_default_graph()
model = hybrid_model()
model.train(data_generator)
#model.train(data_generator, test_generator)
