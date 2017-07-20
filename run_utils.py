import subprocess, yaml
from hybrid_model import HybridModel, create_local_log_path, generate_multi_model_tensorboard_script


def run_jobs_in_parallel(job_commands, model_name_list, log_process=True, model_set_name=None):
    processes = set()
    log_path_dict = {}
    for command_line, model_name in zip(job_commands, model_name_list):
        local_log_path = create_local_log_path(HybridModel.COMMON_PATH, model_name)
        log_path_dict[model_name] = local_log_path
        print "running process:", " ".join(command_line)
        if log_process:
            log_file = "{}_log.txt".format(model_name)
            with open(log_file, 'w') as output:
                processes.add(subprocess.Popen(command_line, stdout=output, bufsize=0))
        else:
            processes.add(subprocess.Popen(command_line))

    if log_process and model_set_name:
        with open("{}_log_path.yaml".format(model_set_name), "w") as output:
            yaml.dump(log_path_dict, output)
    else:
        generate_multi_model_tensorboard_script(log_path_dict)
    for p in processes:
        if p.poll() is None:
            p.wait()