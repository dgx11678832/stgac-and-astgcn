import argparse
import sys

import pandas as pd
import os
import pathlib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import scipy.sparse as sp
from utils import *
from NET import TrafficModel, TrafficModelAstgcn
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parse = argparse.ArgumentParser(description='Calculate cylinder volume')
    parse.add_argument('--kernel_size',  default=3, type=int, help='stgcn 2Dcov kernerl_size')
    parse.add_argument('--K', type=int, default=3, help=" Chebyshev filter size")
    parse.add_argument('--learning_rate', type=float, default=0.01,help='学习率')
    parse.add_argument('--batch_size', type=int, default=15)
    parse.add_argument('--num_epochs', type=int, default=4)
    parse.add_argument('--num_layers', type=int, default=1,help='stgcn :需要调整n_his  astgcn:重复次数 ')
    parse.add_argument('--n_his', type=int, default=8,help='几个序列为一组')
    parse.add_argument('--n_pred', type=int, default=5,help='要预测接下来几个序列')
    parse.add_argument('--train_prop', type=float, default=0.8,help='训练集占全部比例 训练集测试集验证集相加小于等于1')
    parse.add_argument('--val_prop', type=float, default=0.1)
    parse.add_argument('--test_prop', type=float, default=0.1)
    parse.add_argument('--model_save_path', type=str, default='best_model_.pt',help='模型保存位置')
    parse.add_argument('--weighted_adj_matrix_path', type=str, default='data/distances.csv')
    parse.add_argument('--feature_vectors_path', type=str, default='data/combined.csv')
    parse.add_argument('--save_txt', type=bool, default=True)
    parse.add_argument('--model', type=str, default='astgcn',help='astgcn or stgcn')
    parse.add_argument('--nb_chev_filter', type=int, default=4,help='切比雪夫滤波器的数量')
    parse.add_argument('--nb_time_filter', type=int, default=8,help='时间过滤器的数量')
    parse.add_argument('--time_strides', type=int, default=1,help='时空卷积过程中的时间跨度')
    parse.add_argument('--visualization', type=bool, default=True)
    parse.add_argument('--lineplot', type=bool, default=True)
    parse.add_argument('--histplot0', type=bool, default=True)
    parse.add_argument('--histplot1', type=bool, default=True)
    parse.add_argument('--show', type=bool, default=True)
    # parse.add_argument('--model_save_path', type=str, default='./best_model.pt')
    args = parse.parse_args()
    return args

def read_data(device):
    root_path = str(pathlib.Path.cwd())
    W = pd.read_csv(os.path.join(root_path, args.weighted_adj_matrix_path)).drop('station_id', axis=1)
    V = pd.read_csv(os.path.join(root_path, args.feature_vectors_path)).drop('time', axis=1)
    for pos, col in enumerate(W.columns):
        if col != V.columns[pos]:
            print(col, V.columns[pos])
    num_samples, num_nodes = V.shape
    len_train = round(num_samples * args.train_prop)
    len_val = round(num_samples * args.val_prop)
    train = V[: len_train]
    val = V[len_train: len_train + len_val]
    test = V[len_train + len_val: len_train + len_val + round(num_samples * args.test_prop)]

    scaler = StandardScaler()
    train = np.nan_to_num(scaler.fit_transform(train)) # 找出train的均值和标准差，并应用在train上 、将空改为0
    val = np.nan_to_num(scaler.transform(val))
    test = np.nan_to_num(scaler.transform(test))

    x_train, y_train = data_transform(train, args.n_his, args.n_pred, device) #时间序列预测n_his表示几个为一组，n_pred要预测多久
    x_val, y_val = data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = data_transform(test, args.n_his, args.n_pred, device)
    return x_train, y_train, x_val, y_val, x_test, y_test, num_samples, num_nodes, scaler, V, W


def create_dataset(x_train, y_train, x_val, y_val, x_test, y_test, W,device):
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    val_iter = torch.utils.data.DataLoader(val_data, args.batch_size)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    test_iter = torch.utils.data.DataLoader(test_data, args.batch_size)
    G = sp.coo_matrix(W) # 创建矩阵
    edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64).to(device)
    edge_weight = torch.tensor(G.data).float().to(device)
    edge_index = edge_index[:, :100000] # 显存不足 截取一部分
    edge_weight = edge_weight[:100000]
    return train_iter, val_iter, test_iter, edge_index, edge_weight

def create_model(num_nodes, model, device):
    if model == "stgcn":
        model_ = TrafficModel(device, num_nodes, channels, args.num_layers, args.kernel_size, args.K, \
                          args.n_his, normalization='sym', bias=True).to(device)
        # device 驱动, num_nodes 节点个数, channels 通道数 ，args 的在上面解释了
    elif model == "astgcn":
        model_ = TrafficModelAstgcn(device, num_nodes, channels, args.num_layers, args.K, args.n_his, \
                                    args.nb_chev_filter, args.nb_time_filter, args.time_strides,\
                                    normalization='sym',bias=True).to(device)
    else:
        raise ValueError("Without this network")
    return model_


def train(model, train_iter, edge_index, edge_weight, device, val_iter):
    loss = nn.MSELoss() #损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) #优化器
    min_val_loss = np.inf
    for epoch in tqdm(range(1, args.num_epochs + 1), desc='Epoch', position=0):
        l_sum, n = 0.0, 0

        model.train()

        for x, y in tqdm(train_iter, desc='Batch', position=0):
            if args.model == "astgcn":
                x = x.permute(0, 2, 3, 1) # B T_in Num_node F_in -> B Nun_node F_in T_in  // stg -> astg
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
            torch.save(model.state_dict(), args.model_save_path)
        print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
def eval(model, test_iter, edge_index, edge_weight, scaler, V, device):
    loss = nn.MSELoss()
    model.load_state_dict(torch.load(args.model_save_path))
    l = evaluate_model(model, loss, test_iter, edge_index, edge_weight, device)
    MAE, MAPE, RMSE = evaluate_metric(model, test_iter, scaler, edge_index, edge_weight, device)
    print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
    if args.save_txt:
        predict_model = model

        pred_save_path = os.path.join(
            str(pathlib.Path.cwd()), 'results', args.model + '.csv')
        # pred_save_path = os.path.join(
        #     str(pathlib.Path.cwd()), 'results', args.model + args.num_epochs + '.csv')
        pred_len = 144

        pred_set = V[(-pred_len - args.n_his - args.n_pred):]
        pred_set = np.nan_to_num(scaler.transform(pred_set))
        x_pred, y_pred = data_transform(pred_set, args.n_his, args.n_pred, device)
        pred_data = torch.utils.data.TensorDataset(x_pred, y_pred)
        pred_iter = torch.utils.data.DataLoader(pred_data, pred_len + args.n_his + args.n_pred + 1)

        predictions = get_predictions(predict_model, pred_iter, scaler, edge_index, edge_weight, 1475, device)
        np.savetxt(pred_save_path, predictions[1], delimiter=',')
    if args.visualization:
        visualization(pred_save_path, pred_len, V)

def visualization(pred_save_path, pred_len, V):
    # plt.rcParams["figure.figsize"] = (15, 10)
    if args.lineplot:
        test_preds = pd.read_csv(pred_save_path, header=None)
        test_truth = V[(-pred_len - 2):-2]
        summed_test_truth = test_truth.mean(axis=1)
        summed_test_preds = test_preds.mean(axis=1)
        sns.lineplot(y=summed_test_truth, x=range(len(summed_test_preds)))
        sns.lineplot(y=summed_test_preds, x=range(len(summed_test_preds)))
        plt.legend(labels=['Ground Truth', 'Prediction'])
        plt.savefig("{}.png".format(pred_save_path.split('.')[0] + "_lineplot"))
        if args.show:
            plt.show()
    if args.histplot0:
        test_preds = pd.read_csv(pred_save_path, header=None)
        test_truth = V[(-pred_len - 2):-2]
        summed_test_truth = test_truth.mean(axis=0)
        summed_test_preds = test_preds.mean(axis=0)
        sns.histplot(summed_test_truth)
        sns.histplot(summed_test_preds, color=sns.color_palette()[1])
        plt.legend(labels=['Ground Truth', 'Prediction'])
        plt.savefig("{}.png".format(pred_save_path.split('.')[0] + '_histplot0'))
        if args.show:
            plt.show()
    if args.histplot0:
        test_preds = pd.read_csv(pred_save_path, header=None)
        test_truth = V[(-pred_len - 2):-2]
        summed_test_truth = test_truth.mean(axis=1)
        summed_test_preds = test_preds.mean(axis=1)
        sns.histplot(summed_test_truth)
        sns.histplot(summed_test_preds, color=sns.color_palette()[1])
        plt.legend(labels=['Ground Truth', 'Prediction'])
        plt.savefig("{}.png".format(pred_save_path.split('.')[0] + '_histplot1'))
        if args.show:
            plt.show()


if __name__ == '__main__':
    args = parse_args()
    args.model_save_path = args.model_save_path.split('.')[-2] + args.model + '.' + args.model_save_path.split('.')[-1]
    channels = np.array([[1, 4, 8], [8, 4, 8]])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    x_train, y_train, x_val, y_val, x_test, y_test, num_samples, num_nodes, scaler, V, W = read_data(device)
    train_iter, val_iter, test_iter, edge_index, edge_weight = create_dataset(x_train, y_train, x_val, y_val, x_test, y_test, W,device)
    model = create_model(num_nodes, args.model, device)
    train(model, train_iter, edge_index, edge_weight, device, val_iter)
    eval(model, test_iter, edge_index, edge_weight, scaler, V, device)

