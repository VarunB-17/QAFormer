import argparse


def config():
    arguments = argparse.ArgumentParser(
        prog='Batch config',
        description='pytorch dataloader parameters'
    )
    # pre-processing arguments
    arguments.add_argument("-cl", "--context_len",
                           help='maximum context length',
                           default=300, type=int)
    arguments.add_argument("-wl", "--word_len",
                           help='maximum word length',
                           default=16, type=int)
    arguments.add_argument("-bz", "--batch_size",
                           help='batch size',
                           default=32, type=int)
    arguments.add_argument("-s", "--save",
                           help='save',
                           default=True, type=bool)
    arguments.add_argument("-mq", "--maxq",
                           help='max answer length',
                           default=30, type=int)
    arguments.add_argument("-ql", "--question_len",
                           help='max question length',
                           default=40, type=int)

    cfg = arguments.parse_args()
    return cfg


_params = config()

if __name__ == '__main__':
    print(config())



