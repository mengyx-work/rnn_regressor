import os, sys, yaml, argparse
from hybrid_model import generate_multi_model_tensorboard_script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', help='use all the yaml files', action='store_true')
    args = parser.parse_args()
    if args.all:
        mylist = os.listdir(os.getcwd())
        file_list = [file_name for file_name in mylist if ".yaml" in file_name]
    else:
        file_list = sys.argv[1:]
    model_dict = {}
    for file_name in file_list:
        print "file name: {}".format(file_name)
        with open(file_name, 'r') as yaml_file:
            single_model_dict = yaml.load(yaml_file)
        model_dict.update(single_model_dict)
    generate_multi_model_tensorboard_script(model_dict)

if __name__ == "__main__":
    main()
