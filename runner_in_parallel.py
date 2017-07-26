from run_utils import run_jobs_in_parallel

#common_command_line = ["python", "runner.py"]
common_command_line = ["python", "multi_fold_runner.py"]
learning_rate_list = [0.01, 0.001, 0.0001]
batch_size_list = [100, 500, 1000]
job_command_list = []
model_name_list = []
GCS_path = 'test/ML'
yaml_GCS_path = 'test/ML/index_yaml'
#yaml_file_name = 'training_configuration.yaml'
yaml_file_name = 'processed_data_configuration.yaml'

for batch_size in batch_size_list:
    for learning_rate in learning_rate_list:
        model_name = "learning_rate_{}_batch_size_{}".format(learning_rate, batch_size)
        args = {}
        args['--model_name'] = model_name
        args['--learning_rate'] = str(learning_rate)
        args['--batch_size'] = str(batch_size)

        args['--gcs_path'] = GCS_path
        args['--yaml_file_name'] = yaml_file_name
        args['--index_gcs_path'] = yaml_GCS_path
        args['--fold_num'] = str(1)

        command_lines = common_command_line[:]
        command_lines.extend(reduce(lambda x, y: x + y, args.items()))
        model_name_list.append(model_name)
        job_command_list.append(command_lines)

print "running {} jobs".format(len(job_command_list))
run_jobs_in_parallel(job_command_list, model_name_list, False)
