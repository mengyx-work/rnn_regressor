import os, argparse
from series_data_generator import SeriesDataGenerator
from hybrid_model import HybridModel, model_predict, create_local_model_path
from utils import load_training_data_from_gcs
from data_preprocess import create_train_valid_data

GCS_path = 'test/ML'
yaml_file_name = 'training_configuration.yaml'
fraction = 0.7
config_dict, local_data_file = load_training_data_from_gcs(GCS_path, yaml_file_name)
train, valid_data = create_train_valid_data(local_data_file, config_dict.copy(), fraction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning_rate", help="model training learning rate", type=float, default=0.00001)
    parser.add_argument("-b", "--batch_size", help="model training batch size", type=int, default=50)
    parser.add_argument("-n", "--model_name", help="model name, also the folder name", type=str, default="NYDN_hybrid_model")
    args = parser.parse_args()

    data_generator = SeriesDataGenerator(train, config_dict)
    test_generator = SeriesDataGenerator(valid_data, config_dict)
    try:
        #tf.reset_default_graph()
        print "learning rate: {}, batch size: {}".format(args.learning_rate, args.batch_size)
        model = HybridModel(config_dict, args.model_name, learning_rate=args.learning_rate, batch_size=args.batch_size)
        model.train(data_generator, test_generator)
    except Exception, e:
        print "found the exception {} in model training.".format(e)
    finally:
        os.unlink(local_data_file)

    '''
    pred_op_name = "NYDN_hybrid_model/fully_connect_layer/fully_connect_layer_2/Relu:0"
    local_model_path = create_local_model_path(HybridModel.COMMON_PATH, model_name)
    combined_results = model_predict(valid_data, config_dict.copy(), local_model_path, pred_op_name)
    combined_results.to_csv("../combined_pred_results.csv", header=True)
    '''

if __name__ == "__main__":
    main()