import subprocess
from hybrid_model import HybridModel, create_local_log_path, generate_multi_model_tensorboard_script

common_command_line = ["python", "runner.py"]
learning_rate_list = [0.00001, 0.0001, 0.001, 0.01]
batch_size_list = [100]
job_command_list = []
model_name_list = []


def run_jobs_in_parallel(job_commands, model_name_list):
    processes = set()
    log_path_dict = {}
    for command_line, model_name in zip(job_commands, model_name_list):
        local_log_path = create_local_log_path(HybridModel.COMMON_PATH, model_name)
        log_path_dict[model_name] = local_log_path
        log_file = "{}_log.txt".format(model_name)
        print "running process:", " ".join(command_line)
        with open(log_file, 'w') as output:
            processes.add(subprocess.Popen(command_line, stdout=output, bufsize=0))

    generate_multi_model_tensorboard_script(log_path_dict)
    for p in processes:
        if p.poll() is None:
            p.wait()


for learning_rate in learning_rate_list:
    for batch_size in batch_size_list:
        model_name = "model_learning_rate_{}_batch_size_{}".format(learning_rate, batch_size)
        args = {}
        args['--model_name'] = model_name
        args['--learning_rate'] = str(learning_rate)
        args['--batch_size'] = str(batch_size)
        command_lines = common_command_line[:]
        command_lines.extend(reduce(lambda x, y: x + y, args.items()))
        model_name_list.append(model_name)
        job_command_list.append(command_lines)

print len(job_command_list)
run_jobs_in_parallel(job_command_list, model_name_list)