import os, subprocess, stat
from hybrid_model import HybridModel, create_local_log_path

common_command_line = ["python", "runner.py"]
learning_rate_list = [0.00001]
batch_size_list = [1, 10, 50, 100]
job_command_list = []
model_name_list = []


def generate_multi_model_tensorboard_script(log_path_dict):
    file_name = "start_multi_model_tensorboard.sh"
    logdir = ",".join(["{}:{}".format(name, model_log_path) for name, model_log_path in log_path_dict.iteritems()])
    with open(file_name, "w") as text_file:
        text_file.write("#!/bin/bash \n")
        text_file.write("tensorboard --logdir={}".format(logdir))
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IEXEC)


def run_jobs_in_parallel(job_commands, model_name_list):
    processes = set()
    log_path_dict = {}
    for command_lines, model_name in zip(job_commands, model_name_list):
        local_log_path = create_local_log_path(HybridModel.COMMON_PATH, model_name)
        log_path_dict[model_name] = local_log_path
        log_file = "{}_log.txt".format(model_name)
        print "running process:", " ".join(command_lines)
        with open(log_file, 'w') as output:
            processes.add(subprocess.Popen(command_lines, stdout=output, bufsize=0))

    generate_multi_model_tensorboard_script(log_path_dict)
    for p in processes:
        if p.poll() is None:
            p.wait()


for learning_rate in learning_rate_list:
    for batch_size in batch_size_list:
        model_name = "NYDN_hybrid_model_learning_rate_{}_batch_size_{}".format(learning_rate, batch_size)
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