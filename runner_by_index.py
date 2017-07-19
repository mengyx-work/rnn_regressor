from data_preprocess import create_kfold_data_index_yaml_files
from run_utils import run_jobs_in_parallel

GCS_path = 'test/ML'
yaml_GCS_path = 'test/ML/index_yaml'
yaml_file_name = 'training_configuration.yaml'
model_name = 'NYDN_hybrid_model'
fold_num = 2
yaml_file_list, model_name_list = create_kfold_data_index_yaml_files(GCS_path,
                                                                     yaml_file_name,
                                                                     yaml_GCS_path,
                                                                     model_name,
                                                                     fold_num)
print yaml_file_list, model_name_list
common_command_line = ["python", "single_runner_by_index.py"]
job_command_list = []
for yaml_file, model_name in zip(yaml_file_list, model_name_list):
    args = {}
    args['--model_name'] = model_name
    args['--index_gcs_path'] = yaml_GCS_path
    args['--index_file_name'] = yaml_file
    command_lines = common_command_line[:]
    command_lines.extend(reduce(lambda x, y: x + y, args.items()))
    print command_lines
    model_name_list.append(model_name)
    job_command_list.append(command_lines)

print len(job_command_list)
run_jobs_in_parallel(job_command_list, model_name_list)

