import os, argparse
from series_data_generator import SeriesDataGenerator
from google_cloud_storage_util import GCS_Bucket
from hybrid_model import HybridModel
from data_preprocess import load_training_data_from_gcs, load_yaml_file_from_gcs, create_train_test_by_index


def main():
    '''entry point for TensorFlow model training, this model training
    requires a yaml file in GCS to provide the set of `train_index` and
    `test_index` used in function `create_train_test_by_index`.

    command-line entry points
        1. --learning_rate: the learning rate used in model training
        2. --batch_size: the batch size used in model training
        3. --model_name: the model name used in model training

        4. --gcs_path: GCS path for the model configuration yaml file
        5. --yaml_file_name: yaml configuration file name in GCS
        6. --index_gcs_path: GCS path for the data index yaml file
        7. --index_file_name: index yaml file name in GCS
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning_rate", help="model training learning rate", type=float, default=0.00005)
    parser.add_argument("-b", "--batch_size", help="model training batch size", type=int, default=0)
    parser.add_argument("-n", "--model_name", help="model name, also the folder name", type=str, default="NYDN_hybrid_model")
    parser.add_argument("--gcs_path", help="the GCS path for config_dict and data", type=str, default="test/ML")
    parser.add_argument("--yaml_file_name", help="model name, also the folder name", type=str, default="target_median_norm_configuration.yaml")
    parser.add_argument("--index_gcs_path", help="the GCS path for config_dict and data", type=str, default="test/ML/index_yaml")
    #parser.add_argument("--index_file_name", help="model name, also the folder name", type=str, default="target_median_last_hid8_16-1_learning_rate_0.001_MAE_fold_1.yaml")
    parser.add_argument("--index_file_name", help="model name, also the folder name", type=str)

    args = parser.parse_args()
    config_dict, local_data_file = load_training_data_from_gcs(args.gcs_path, args.yaml_file_name)
    bucket = GCS_Bucket()
    index_dict = load_yaml_file_from_gcs(bucket, args.index_gcs_path, args.index_file_name)
    train, valid_data = create_train_test_by_index(local_data_file, config_dict, index_dict)
    data_generator = SeriesDataGenerator(train, config_dict)
    test_generator = SeriesDataGenerator(valid_data, config_dict)
    try:
        #tf.reset_default_graph()
        #print "learning rate: {}, batch size: {}".format(args.learning_rate, args.batch_size)
        model = HybridModel(config_dict,
                            args.model_name,
                            learning_rate=args.learning_rate,
                            batch_size=data_generator.get_total_counts())
        model.train(data_generator, test_generator)
    except Exception, e:
        print "found the exception {} in model training.".format(e)
    finally:
        os.unlink(local_data_file)

if __name__ == "__main__":
    main()
