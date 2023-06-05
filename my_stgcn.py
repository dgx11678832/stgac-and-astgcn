#%%
from datetime import datetime
import geopy.distance
import glob
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn import STConv
from tqdm import tqdm
from torch_geometric_temporal.nn import ASTGCN
from torch_geometric.nn import GCNConv

class FullyConnLayer(nn.Module):
    def __init__(self, c):
        super(FullyConnLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)
        self.linear = nn.Linear(c,1)
    def forward(self, x):
        # return self.conv(x)
        return self.linear(x)
class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        # self.tconv1 = nn.Conv2d(4, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.conv1 = GCNConv(8, 16)
        self.ln = nn.LayerNorm([n, 16])
        # self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.conv2 = GCNConv(16, 8)
        self.fc = FullyConnLayer(c)

    def forward(self, x,edge):
        x_t1 = self.conv1(x,edge)
        x_ln = self.ln(x_t1)
        x_t2 = self.conv2(x_ln,edge)

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
class TrafficModel2(torch.nn.Module):
    def __init__(self, device, num_nodes, channel_size_list, num_layers,num_samples,
                 kernel_size, K, window_size, \
                 normalization = 'sym', bias = True):

        super(TrafficModel2, self).__init__()
        self.layers = nn.ModuleList([])
        input_size, hidden_size, output_size = channel_size_list[0][0], channel_size_list[0][1], \
        channel_size_list[0][2]
        self.layers.append(ASTGCN(nb_block=num_layers,in_channels=input_size,\
                                  K=K,nb_chev_filter=4,nb_time_filter=8,\
                                  time_strides=1,num_for_predict=output_size,\
                                  len_input=8,num_of_vertices=num_nodes,
                                  normalization=normalization
                                  ))
        self.layers.append(OutputLayer(channel_size_list[0][-1], \
                                       window_size - 2 * num_layers * (kernel_size - 1), \
                                       num_nodes))
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers[:-1]:
          x = layer(x, edge_index)

        out_layer = self.layers[-1]
        # x = x.permute
        x = out_layer(x, edge_index)
        return x

def data_transform(data, n_his, n_pred, device):
    num_nodes = data.shape[1]
    num_obs = len(data) - n_his - n_pred
    x = np.zeros([num_obs, n_his, num_nodes, 1])
    y = np.zeros([num_obs, num_nodes])

    obs_idx = 0
    for i in range(num_obs):
        head = i
        tail = i + n_his
        x[obs_idx, :, :, :] = data[head: tail].reshape(n_his, num_nodes, 1)
        y[obs_idx] = data[tail + n_pred - 1]
        obs_idx += 1

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def evaluate_model(model, loss, data_iter, edge_index, edge_weight, device):
  model.eval()
  l_sum, n = 0.0, 0
  with torch.no_grad():
      for x1, y in data_iter:
          x = x1.permute(0, 2, 3, 1)
          y_pred = model(x.to(device), edge_index, edge_weight).view(len(x), -1)
          l = loss(y_pred, y)
          l_sum += l.item() * y.shape[0]
          n += y.shape[0]
      return l_sum / n

def evaluate_metric(model, data_iter, scaler, edge_index, edge_weight, device):
    model.eval()
    epsilon = 1e-6
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x1, y in data_iter:
            x = x1.permute(0, 2, 3, 1)
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x.to(device), \
                                                    edge_index, \
                                                    edge_weight).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / (y+epsilon)).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE


def get_predictions(model, pred_iter, scaler, edge_index, edge_weight, num_nodes, device):
    model.eval()
    with torch.no_grad():
        for x1, y in pred_iter:
            x = x1.permute(0, 2, 3, 1)
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x.to(device), \
                                                    edge_index, \
                                                    edge_weight).view(len(x), -1).cpu().numpy()).reshape(-1)
            y_pred = y_pred.reshape(-1, num_nodes)

        return y, y_pred

channels = np.array([[1, 4, 8], [8, 4, 8]])

kernel_size = 3
K = 3

learning_rate = 0.01
batch_size = 10
num_epochs = 4 #20
num_layers = 1
n_his = 8

n_pred = 5

train_prop = 0.1
val_prop = 0.1
test_prop = 0.1

# model_save_path = os.path.join(
#     'best_model.pt')
model_save_path = './best_model.pt'
weighted_adj_matrix_path = os.path.join('/home/deepblue/PycharmProjects/STGCN+/GBDS_Project-main+(1)/GBDS_Project-main/processed_data/distances.csv')
W = pd.read_csv(weighted_adj_matrix_path).drop('station_id', axis=1)

feature_vectors_path = os.path.join('/home/deepblue/PycharmProjects/STGCN+/GBDS_Project-main+(1)/GBDS_Project-main/processed_data/combined.csv')
V = pd.read_csv(feature_vectors_path).drop('time', axis=1)

for pos, col in enumerate(W.columns):
  if col != V.columns[pos]:
    print(col, V.columns[pos])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_samples, num_nodes = V.shape

len_train = round(num_samples * train_prop)
len_val = round(num_samples * val_prop)
train = V[ : len_train]
val = V[len_train : len_train + len_val]
test = V[len_train + len_val : len_train + len_val + round(num_samples * test_prop)]

scaler = StandardScaler()
train = np.nan_to_num(scaler.fit_transform(train))
val = np.nan_to_num(scaler.transform(val))
test = np.nan_to_num(scaler.transform(test))

x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)

G = sp.coo_matrix(W)
edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64).to(device)
edge_weight = torch.tensor(G.data).float().to(device)
edge_index = edge_index[:,:100000]
edge_weight = edge_weight[:100000]

model2 = TrafficModel(device, num_nodes, channels, num_layers, kernel_size, K, \
                     n_his, normalization = 'sym', bias = True).to(device)
model = TrafficModel2(device, num_nodes, channels, num_layers, kernel_size, K, num_samples,\
                     n_his, normalization = 'sym', bias = True).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
min_val_loss = np.inf
for epoch in tqdm(range(1, num_epochs + 1), desc = 'Epoch', position = 0):
  l_sum, n = 0.0, 0

  model.train()

  for x1, y in tqdm(train_iter, desc = 'Batch', position = 0):
    x = x1.permute(0, 2, 3, 1)
    y_pred = model(x.to(device), edge_index, edge_weight).view(len(x), -1)
    l = loss(y_pred, y)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    l_sum += l.item() * y.shape[0]
    n += y.shape[0]

  val_loss = evaluate_model(model, loss, val_iter, edge_index, edge_weight, device)
  if val_loss < min_val_loss:
      min_val_loss = val_loss
      torch.save(model.state_dict(), model_save_path)
  print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
best_model = TrafficModel2(device, num_nodes, channels, num_layers, kernel_size, K, num_samples,\
                     n_his, normalization = 'sym', bias = True).to(device)
best_model.load_state_dict(torch.load(model_save_path))

l = evaluate_model(best_model, loss, test_iter, edge_index, edge_weight, device)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler, edge_index, edge_weight, device)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)

predict_model = best_model

pred_save_path = os.path.join(
    '/home/deepblue/PycharmProjects/STGCN+',  'predictions_testing.csv')

pred_len = 144

pred_set = V[(-pred_len - n_his - n_pred):]
pred_set = np.nan_to_num(scaler.transform(pred_set))
x_pred, y_pred = data_transform(pred_set, n_his, n_pred, device)
pred_data = torch.utils.data.TensorDataset(x_pred, y_pred)
pred_iter = torch.utils.data.DataLoader(pred_data, pred_len + n_his + n_pred + 1)

predictions = get_predictions(predict_model, pred_iter, scaler, edge_index, edge_weight, 1475, device)
np.savetxt(pred_save_path, predictions[1], delimiter=',')



import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,10)
test_preds = pd.read_csv(pred_save_path, header=None)
test_truth = V[(-pred_len-2):-2]
summed_test_truth  = test_truth.mean(axis=1)
summed_test_preds = test_preds.mean(axis=1)
sns.lineplot(y=summed_test_truth, x=range(len(summed_test_preds)))
sns.lineplot(y=summed_test_preds, x=range(len(summed_test_preds)))
plt.legend(labels=['Ground Truth', 'Prediction']);
plt.show()
#%%
test_preds = pd.read_csv(pred_save_path, header=None)
test_truth = V[(-pred_len-2):-2]
summed_test_truth  = test_truth.mean(axis=1)
summed_test_preds = test_preds.mean(axis=1)
sns.histplot(summed_test_truth)
sns.histplot(summed_test_preds, color=sns.color_palette()[1])
plt.legend(labels=['Ground Truth', 'Prediction'])

plt.legend(labels=['Ground Truth', 'Prediction']);
#%%
test_preds = pd.read_csv(pred_save_path, header=None)
test_truth = V[(-pred_len-2):-2]
summed_test_truth  = test_truth.mean(axis=0)
summed_test_preds = test_preds.mean(axis=0)
sns.histplot(summed_test_truth)
sns.histplot(summed_test_preds, color=sns.color_palette()[1])
plt.legend(labels=['Ground Truth', 'Prediction']);
