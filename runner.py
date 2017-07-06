from series_data_generator import SeriesDataGenerator
from hybrid_model import hybrid_model
from utils import create_column_config
from data_preprocess import create_train_valid_data

data_path = '/Users/matt.meng/taboola_data/taboola_process_data'
processed_train_file = 'NYDN_240min_fullWindow_60min_exposure_120seconds_interval_target_24hr_data.csv'
index_col = "articleId"
fraction = 0.7
config = create_column_config()
train, valid_data = create_train_valid_data(data_path, processed_train_file, config.target_column, index_col, fraction)

data_generator = SeriesDataGenerator(train, config)
test_generator = SeriesDataGenerator(valid_data, config)

#tf.reset_default_graph()
model = hybrid_model()
#model.train(data_generator)
model.train(data_generator, test_generator)

