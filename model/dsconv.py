from torch import nn

class DSConv(nn.Module):
    def __init__(self, kernel_size, input_dim, out_dim, bias=False):
        """""
        Depth-wise separable convolution behave similar 
        like regular convolution but requires less 
        computation for the exact same operations such
        that less parameters are needed in the computation
        process.
        """""
        super().__init__()
        self.depth = nn.Conv1d(in_channels=input_dim,
                               out_channels=input_dim,
                               groups=input_dim,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2,
                               bias=bias)

        self.point = nn.Conv1d(in_channels=input_dim,
                               out_channels=out_dim,
                               kernel_size=1,
                               padding=0,
                               bias=bias)

        # self.normal_conv = nn.Conv1d(input_dim, out_dim, kernel_size=5, padding=kernel_size // 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = self.normal_conv(x)
        x = self.depth(x)
        x = self.point(x)
        x = x.permute(0, 2, 1)
        return x
