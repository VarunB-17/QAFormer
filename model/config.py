import argparse


def config():
    arguments = argparse.ArgumentParser(
        prog='Batch config',
        description='pytorch dataloader parameters'
    )
    # pre-processing arguments
    arguments.add_argument("-wp", "--word_path",
                           help='pre-trained word embedding path',
                           default='..\\dataloader\\data\\word_emb.pt')
    arguments.add_argument("-cp", "--char_path",
                           help='pre-trained character embedding path',
                           default='..\\dataloader\\data\\char_emb.pt')
    arguments.add_argument("-md", "--model_dim",
                           help='model output dimension',
                           default=128, type=int)
    arguments.add_argument("-ce", "--char_emb",
                           help='character embedding dimension',
                           default=200, type=int)
    arguments.add_argument("-we", "--word_emb",
                           help='character embedding dimension',
                           default=300, type=int)
    arguments.add_argument("-cd", "--c_drop",
                           help='dropout probability for character embedding',
                           default=0.2, type=float)
    arguments.add_argument("-wd", "--w_drop",
                           help='dropout probability for word embedding',
                           default=0.3, type=float)
    arguments.add_argument("-wl", "--word_length",
                           help='amount of characters for each word',
                           default=16, type=int)
    arguments.add_argument("-id", "--input_dim",
                           help='model dimension',
                           default=500, type=int)
    arguments.add_argument("-hl", "--highway-layer",
                           help='amount of highway layers in the network',
                           default=2, type=int)



    cfg = arguments.parse_args()
    return cfg


_modelcfg = config()

if __name__ == '__main__':
    print(config())
