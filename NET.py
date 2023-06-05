import torch
import torch.nn as nn

from torch_geometric_temporal.nn import ASTGCN
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn import STConv

class FullyConnLayer2(nn.Module):
    def __init__(self, c):
        super(FullyConnLayer2, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)
        self.linear = nn.Linear(c,1)
    def forward(self, x):
        # return self.conv(x)
        return self.linear(x)
class OutputLayer2(nn.Module):
    def __init__(self, c, n):
        super(OutputLayer2, self).__init__()
        # self.tconv1 = nn.Conv2d(4, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.conv1 = GCNConv(8, 16)
        self.ln = nn.LayerNorm([n, 16])
        # self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.conv2 = GCNConv(16, 8)
        self.fc = FullyConnLayer2(c)

    def forward(self, x,edge, edge_weight):
        x_t1 = self.conv1(x,edge, edge_weight)
        x_ln = self.ln(x_t1)
        x_t2 = self.conv2(x_ln,edge, edge_weight)

        return self.fc(x_t2)
class TrafficModelAstgcn(torch.nn.Module):
    def __init__(self, device, num_nodes, channel_size_list, num_layers, k, n_hid, \
                 nb_chev_filter,nb_time_filter,time_strides,
                 normalization='sym', bias = True):

        super(TrafficModelAstgcn, self).__init__()
        self.layers = nn.ModuleList([])
        input_size, hidden_size, output_size = channel_size_list[0][0], channel_size_list[0][1], \
        channel_size_list[0][2]
        self.layers.append(ASTGCN(nb_block=num_layers,in_channels=input_size,\
                                  K=k,nb_chev_filter=nb_chev_filter,nb_time_filter=nb_time_filter,\
                                  time_strides=time_strides,num_for_predict=output_size,\
                                  len_input=n_hid,num_of_vertices=num_nodes,
                                  normalization=normalization,bias=bias,
                                  ))
        self.layers.append(OutputLayer2(channel_size_list[0][-1], num_nodes))

        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers[:-1]:
          x = layer(x, edge_index)

        out_layer = self.layers[-1]
        # x = x.permute
        x = out_layer(x, edge_index, edge_weight)
        return x
# class TrafficModelAstgcn(torch.nn.Module):
#     def __init__(self, device, num_nodes, channel_size_list, num_layers,
#                  kernel_size, K, num_samples,window_size, \
#                  normalization = 'sym', bias = True):
#
#         super(TrafficModelAstgcn, self).__init__()
#         self.layers = nn.ModuleList([])
#         input_size, hidden_size, output_size = channel_size_list[0][0], channel_size_list[0][1], \
#         channel_size_list[0][2]
#         self.layers.append(ASTGCN(nb_block=num_layers,in_channels=input_size,\
#                                   K=K,nb_chev_filter=4,nb_time_filter=8,\
#                                   time_strides=1,num_for_predict=output_size,\
#                                   len_input=8,num_of_vertices=num_nodes,
#                                   normalization=normalization
#                                   ))
#         self.layers.append(OutputLayer2(channel_size_list[0][-1], \
#                                        window_size - 2 * num_layers * (kernel_size - 1), \
#                                        num_nodes))
#         for layer in self.layers:
#             layer = layer.to(device)
#
#     def forward(self, x, edge_index, edge_weight):
#         for layer in self.layers[:-1]:
#           x = layer(x, edge_index)
#
#         out_layer = self.layers[-1]
#         # x = x.permute
#         x = out_layer(x, edge_index)
#         return x


class FullyConnLayer(nn.Module):
    def __init__(self, c):
        super(FullyConnLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.fc = FullyConnLayer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)

        return self.fc(x_t2)

class TrafficModel(torch.nn.Module):
    def __init__(self, device, num_nodes, channel_size_list, num_layers,
                 kernel_size, K, window_size, \
                 normalization = 'sym', bias = True):

        super(TrafficModel, self).__init__()
        self.layers = nn.ModuleList([])
        for l in range(num_layers):
            input_size, hidden_size, output_size = \
            channel_size_list[l][0], channel_size_list[l][1], \
            channel_size_list[l][2]
            self.layers.append(STConv(num_nodes, input_size, hidden_size, \
                                      output_size, kernel_size, K, \
                                      normalization, bias))
        self.layers.append(OutputLayer(channel_size_list[-1][-1], \
                                       window_size - 2 * num_layers * (kernel_size - 1), \
                                       num_nodes))
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers[:-1]:
          x = layer(x, edge_index, edge_weight)
        out_layer = self.layers[-1]
        x = x.permute(0, 3, 1, 2)
        x = out_layer(x)
        return x