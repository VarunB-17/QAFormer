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
    arguments.add_argument("-tp", "--train_path",
                           help='training data',
                           default='..\\dataloader\\data\\train.parquet')
    arguments.add_argument("-tep", "--test_path",
                           help='test data',
                           default='..\\dataloader\\data\\val.parquet')
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
                           default=0.05, type=float)
    arguments.add_argument("-wd", "--w_drop",
                           help='dropout probability for word embedding',
                           default=0.1, type=float)
    arguments.add_argument("-wl", "--word_length",
                           help='amount of characters for each word',
                           default=16, type=int)
    arguments.add_argument("-id", "--input_dim",
                           help='model dimension',
                           default=500, type=int)
    arguments.add_argument("-hl", "--highway-layer",
                           help='amount of highway layers in the network',
                           default=2, type=int)
    arguments.add_argument("-kz", "--kernel_size",
                           help='kernel size for depth-wise separable convolution',
                           default=5, type=int)
    arguments.add_argument("-cdim", "--conv_dim",
                           help='output dimension for depth-wise separable convolution',
                           default=128, type=int)
    arguments.add_argument("-hds", "--heads",
                           help='amount of attention heads',
                           default=8, type=int)
    arguments.add_argument("-cr", "--conv_rep",
                           help='amount of convolution in the embedding encoder block',
                           default=4, type=int)
    arguments.add_argument("-crm", "--conv_rep_model",
                           help='amount of convolution in the model encoder block',
                           default=2, type=int)
    arguments.add_argument("-dp", "--dropout_p",
                           help='dropout probability',
                           default=0.1, type=int)
    arguments.add_argument("-cl", "--context_len",
                           default=300, type=int)
    arguments.add_argument("-ql", "--question_len",
                           default=40, type=int)
    arguments.add_argument("-el", "--enc_layer",
                           help='amount of encoder layers in the model encoding block',
                           default=4, type=int)
    arguments.add_argument("-bz", "--batch_size",
                           help='amount of instance in a batch',
                           default=32, type=int)
    arguments.add_argument("-w", "--workers",
                           help='amount of cpu cores utilized while loading data',
                           default=12, type=int)
    arguments.add_argument("-ep", "--epochs",
                           help='amount of epochs',
                           default=3, type=int)
    arguments.add_argument("-deb", "--debug",
                           help='debug mode',
                           default=2, type=bool)
    arguments.add_argument('-lr', "--learning_rate",
                           default=0.001, type=float)
    arguments.add_argument('-pat', "--patience",
                           default=100, type=int)
    arguments.add_argument('-tb', "--test_batch",
                           default=8, type=int)
    cfg = arguments.parse_args()
    return cfg


_modelcfg = config()

if __name__ == '__main__':
    print(config())
