from torch import nn



class VanillaFC(nn.Module):
    def __init__(self, img_width=28, input_size=1, hidden_size=64, num_of_layers=3, output_size=1, header=True,
                 activation=True):
        super(VanillaFC, self).__init__()
        size_after_conv = img_width ** 2
        self.activation = activation
        self.relu = nn.ReLU()

        self.fc_list = nn.ModuleList()
        for _ in range(num_of_layers - 1):
            self.fc_list += [nn.Linear(size_after_conv, hidden_size)]
            size_after_conv = hidden_size
        self.fc_list += [nn.Linear(size_after_conv, output_size, bias=False)]

        # add softmax layer
        self.head = None
        if header:
            self.head = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for fc in self.fc_list[:-1]:
            x = fc(x)
            if self.activation:
                x = self.relu(x)
        x = self.fc_list[-1](x)
        if self.head is not None:
            x = self.head(x)
        return x



class AlphaModel(VanillaFC):
    def __init__(self, img_width=28, input_channels=1, num_of_layers=3, hidden_size=64,
                 output_size=128, header=True, activation=True):
        super().__init__(img_width=img_width,
                         input_size=input_channels,
                         hidden_size=hidden_size,
                         num_of_layers=num_of_layers,
                         output_size=output_size,
                         header=header,
                         activation=activation)