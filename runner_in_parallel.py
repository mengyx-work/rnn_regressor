from run_utils import run_jobs_in_parallel

common_command_line = ["python", "runner.py"]
learning_rate_list = [0.005, 0.001, 0.0005]
batch_size_list = [10, 50, 100, 200, 500, 1000]
job_command_list = []
model_name_list = []

for batch_size in batch_size_list:
    for learning_rate in learning_rate_list:
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