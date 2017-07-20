import sys, yaml
from hybrid_model import generate_multi_model_tensorboard_script


def main():
    model_dict = {}
    for file_name in sys.argv[1:]:
        with open(file_name, 'r') as yaml_file:
            single_model_dict = yaml.load(yaml_file)
        model_dict.update(single_model_dict)
    generate_multi_model_tensorboard_script(model_dict)

if __name__ == "__main__":
    main()