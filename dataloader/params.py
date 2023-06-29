import argparse
import os

root_folder_path = os.getcwd()


def config():
    arguments = argparse.ArgumentParser(
        prog='QANet',
        description='question and answering transformer'
    )
    # pre-processing arguments
    arguments.add_argument("-gd", "--glove_dim",
                           help='glove vector embedding dimension',
                           default=300, type=int)
    arguments.add_argument("-gp", "--glove_param",
                           help='glove parameter choice',
                           default='840B', type=str)
    arguments.add_argument("-mc", "--max_context",
                           help='maximum length of a context or passage',
                           default=300, type=int)
    arguments.add_argument("-ce", "--char_dim",
                           help='character vector embedding dimension',
                           default=200, type=int)
    arguments.add_argument("-r", "--root",
                           help='root folder',
                           default=root_folder_path, type=str)
    arguments.add_argument("-mq", "--maxq",
                           help='maximum question length',
                           default=30, type=int)

    _params = arguments.parse_args()
    return _params


_params = config()

if __name__ == '__main__':
    print(config())
