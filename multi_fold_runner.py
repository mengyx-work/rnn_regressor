import argparse
from data_preprocess import create_kfold_data_index_yaml_files
from run_utils import run_jobs_in_parallel


def main():
    '''entry point for a multi-fold cross-validation run with
    a set of hyper-parameterts.

    Function `create_kfold_data_index_yaml_files` is used to
    create the multi-fold of index as yaml files and store them
    at GCS.

    With the loop of index yaml files and model names, system command
    lines to run individual model trainings using `single_runner_by_index.py`
    are prepared and fed into `run_jobs_in_parallel`.

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning_rate", help="model training learning rate", type=float, default=0.01)
    parser.add_argument("-b", "--batch_size", help="model training batch size", type=int, default=1)
    parser.add_argument("-n", "--model_name", help="model name, also the folder name", type=str, default="NYDN_new_model")
    parser.add_argument("--gcs_path", help="the GCS path for config_dict and data", type=str, default="test/ML")
    parser.add_argument("--yaml_file_name", help="model name, also the folder name", type=str, default="training_configuration.yaml")
    parser.add_argument("--index_gcs_path", help="the GCS path for config_dict and data", type=str, default="test/ML/index_yaml")
    parser.add_argument("--fold_num", help="fold number for cross-validation", type=int, default=4)
    args = parser.parse_args()
    yaml_file_list, model_name_list = create_kfold_data_index_yaml_files(args.gcs_path,
                                                                         args.yaml_file_name,
                                                                         args.index_gcs_path,
                                                                         args.model_name,
                                                                         args.fold_num)
    common_command_line = ["python", "single_runner_by_index.py"]
    job_command_list = []
    for yaml_file, model_name in zip(yaml_file_list, model_name_list):
        args_dict = {}
        args_dict['--learning_rate'] = str(args.learning_rate)
        args_dict['--batch_size'] = str(args.batch_size)
        args_dict['--gcs_path'] = args.gcs_path
        args_dict['--index_gcs_path'] = args.index_gcs_path
        args_dict['--model_name'] = model_name
        args_dict['--index_file_name'] = yaml_file
        command_lines = common_command_line[:]
        command_lines.extend(reduce(lambda x, y: x + y, args_dict.items()))
        print command_lines
        model_name_list.append(model_name)
        job_command_list.append(command_lines)

    print "start running {} jobs".format(len(job_command_list))
    run_jobs_in_parallel(job_command_list, model_name_list, log_process=True, model_set_name=args.model_name)

if __name__ == '__main__':
    main()
