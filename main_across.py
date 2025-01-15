import argparse
import os
import time

import psutil
import torch
from tools import evaluate_results_nc, EarlyStopping
from data import load_data6,load_data3
import numpy as np
import warnings
from model_across import models
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random


warnings.filterwarnings("ignore")


def main(args):
    g, features, NS, simlar,labels, num_classes, train_idx, val_idx, test_idx = load_data3()
    # g, features, NS, simlar, labels, num_classes, train_idx, val_idx, test_idx = load_data6()
    ns_num = 2
    labels = labels.to(args['device'])
    svm_macro_avg = np.zeros((7,), dtype=np.float64)
    svm_micro_avg = np.zeros((7,), dtype=np.float64)
    nmi_avg = 0
    ari_avg = 0
    print('开始进行训练，重复次数为 {}\n'.format(args['repeat']))
    in_dims = [feature.shape[1] for feature in features]
    model = models(in_dims,args['hidden_units'],num_classes)
    # model = HAN(num_meta_paths=len(g),
    #             ns_num=ns_num,
    #             in_size=features.shape[1],
    #             # in_size=features_dim[0],
    #             hidden_size=args['hidden_units'],
    #             out_size=num_classes,
    #             num_heads=args['num_heads'],
    #             dropout=args['dropout']).to(args['device'])
    g = [graph.to(args['device']) for graph in g]
    early_stopping = EarlyStopping(patience=args['patience'], verbose=True,
                                   save_path='checkpoint/checkpoint_{}.pt'.format('ACM'))  # 提早停止，设置的耐心值为5
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # b = 0
    # a = 0
    for epoch in range(1000):
        # t1 = time.time()
        model.train()

        logits, h = model(features,g, simlar)
        loss = loss_fcn(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        logits, h = model(features,g, simlar)
        print("h.size", h.size())

        # b = b + 1

        val_loss = loss_fcn(logits[val_idx], labels[val_idx])
        test_loss = loss_fcn(logits[test_idx], labels[test_idx])
        print('Epoch{:d}| Train Loss{:.4f}| Val Loss{:.4f}| Test Loss{:.4f}'.format(epoch + 1, loss.item(),
                                                                                      val_loss.item(),
                                                                                      test_loss.item()))
        early_stopping(val_loss.data.item(), model)

        # t2 = time.time()
        # print('当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        # a = a + (t2 - t1)
        # print('当前进程所有时间',a)
        if early_stopping.early_stop:
            print('提前停止训练!')
            break
    # print("平均时间", a / b)

    print('\n进行测试...')
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format('ACM')))
    model.eval()
    logits, h = model(features,g,  simlar)

    # 使用 t-SNE 生成散点图
    # Y = labels[test_idx].cpu().numpy()
    # ml = TSNE(n_components=2)
    # node_pos = ml.fit_transform(logits[test_idx].detach().cpu().numpy())
    # color_idx = {}
    # for i in range(len(logits[test_idx].detach().cpu().numpy())):
    #     color_idx.setdefault(Y[i], [])
    #     color_idx[Y[i]].append(i)
    # for c, idx in color_idx.items():  # c是类型数，idx是索引
    #     if str(c) == '1':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1)
    #     elif str(c) == '2':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1)
    #     elif str(c) == '0':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
    #     elif str(c) == '3':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#006400', s=15, alpha=1)
    # plt.legend()
    # plt.savefig("CAPD-BMHG" + str(args['dataset']) + "分类图" + str('1') + ".png", dpi=1000,bbox_inches='tight')
    # plt.show()

    # 评估结果
    evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(),
                        int(labels.max()) + 1)  # 使用SVM评估节点


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='这是我们基于GAT所构建的HAN模型')
    parser.add_argument('--dataset', default='', help='数据集')
    parser.add_argument('--lr', default=0.004, help='学习率')
    parser.add_argument('--num_heads', default=[8], help='多头注意力数及网络层数')
    parser.add_argument('--hidden_units', default=8, help='隐藏层数（实际隐藏层数：隐藏层数*注意力头数）')
    parser.add_argument('--dropout', default=0.5, help='丢弃率')
    parser.add_argument('--num_epochs', default=2000, help='最大迭代次数')
    parser.add_argument('--weight_decay', default=0.001, help='权重衰减')
    parser.add_argument('--patience', type=int, default=6, help='耐心值')
    parser.add_argument('--device', type=str, default='cpu', help='使用cuda:0或者cpu')
    parser.add_argument('--repeat', type=int, default=1, help='重复训练和测试次数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)
