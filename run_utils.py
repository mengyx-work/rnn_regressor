import subprocess
from hybrid_model import HybridModel, create_local_log_path, generate_multi_model_tensorboard_script


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
