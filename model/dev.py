import torch
def get_device():
    """
    Checks if gpu is available
    """
    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    return dev