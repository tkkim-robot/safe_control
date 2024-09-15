import torch
import torch.nn as nn
import math


class EnsembleLinear(nn.Module):

    def __init__(self, in_features, out_features, ensemble_size, bias=True):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.Tensor(ensemble_size, in_features, out_features)
        if bias:
            self.biases = torch.Tensor(ensemble_size, 1, out_features)
        else:
            self.register_parameter('biases', None)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights:
            w.transpose_(0, 1)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            w.transpose_(0, 1)

        self.weights = nn.Parameter(self.weights)

        if self.biases is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weights[0].T)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.biases, -bound, bound)
            self.biases = nn.Parameter(self.biases)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.repeat(self.ensemble_size, 1, 1)
        return torch.baddbmm(self.biases, input, self.weights)

    def single_forward(self, input, index):
        if len(input.shape) == 2:
            input = input.repeat(1, 1, 1)

        return torch.baddbmm(self.biases[index].unsqueeze(dim=0), input, self.weights[index].unsqueeze(dim=0))
        # referenced pytorch.nn.linear.py at https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        # return F.linear(input, self.weights[index], self.biases[index][0])

    def extra_repr(self) -> str:
        return 'ensemble_size = {}, in_features={}, out_features={}, biases={}'.format(
            self.ensemble_size, self.in_features, self.out_features, self.biases is not None
        )
