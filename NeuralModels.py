
import torch 
import torch.nn as nn # import the nn module
import math
import torch.nn.functional as F
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('mask', mask)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)


class EfficientDAGNN(nn.Module):
    def __init__(self, layers, layer_connections, connection_masks):
        super(EfficientDAGNN, self).__init__()
        self.linears = nn.ModuleDict()

        for k, v in layer_connections.items():
            self.linears[str(k)] = MaskedLinear(sum(len(layers[v_]) for v_ in v), len(layers[k]),torch.cat([torch.from_numpy(connection_masks[k][v_]).float() for v_ in v], dim=0).t())    
        self.activation = nn.ReLU()
        self.layer_connections = layer_connections
  

    def forward(self, x):
        layer_output = [x]
        for k, v in self.layer_connections.items():
            inputs = torch.cat([layer_output[v_] for v_ in v], dim=1)
            layer_output.append(self.activation(self.linears[str(k)](inputs)))

        return layer_output[-1]
