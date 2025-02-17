import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mask = mask  # Binary mask to control connectivity
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

class DAGNN(nn.Module):
    def __init__(self, layers, layer_connections, connection_masks):
        super(DAGNN, self).__init__()

        # Initialize an empty ModuleDict
        self.module_dict = nn.ModuleDict()


        # Iterate and add modules individually
        for k, v in layer_connections.items():
            for v_ in v:
                # Create a unique key for each module
                module_key = f"{v_}_{k}"  # Example: "0_1" for connection from layer 0 to layer 1

                # Add the Linear module to the ModuleDict
                self.module_dict[module_key] = MaskedLinear(len(layers[v_]), len(layers[k]), torch.from_numpy(connection_masks[k][v_]).float().t())

                # Apply masking using a non-in-place operation
                # This creates a new tensor instead of modifying the original in-place
                #self.module_dict[module_key].weight = nn.Parameter(self.module_dict[module_key].weight * torch.from_numpy(connection_masks[k][v_]).float().t())

        # Store other attributes
        self.activation=nn.ReLU()
        self.layers = layers
        self.layer_connections = layer_connections
        self.connection_masks = connection_masks

    def forward(self, x):
        # Initialize the output with the input
        layer_output={}
        layer_output[0]=x
        # layered_connections are sorted starting from layer 1!
        for k,v in self.layer_connections.items():
          out_sum=0
          for v_ in v:
            out_sum+=self.module_dict[f"{v_}_{k}"](layer_output[v_])
          layer_output[k]=self.activation(out_sum)
        return layer_output[len(self.layers)-1]
