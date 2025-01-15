import numpy as np
import scipy
import torch
import dgl
import scipy.sparse as sp
import torch.nn.functional as F

def load_data3(prefix=r'D:\三分类'):
    features_0 = scipy.sparse.load_npz(prefix + '/product_feature.npz').toarray()  # features_0为商品特征 features_1为用户特征
    features_0 = torch.FloatTensor(features_0)
    features = [features_0]

    similarity_matrix = np.load(prefix + "/similarity.npy")
    similarity_matrix = torch.from_numpy(similarity_matrix).type(torch.FloatTensor)
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    num_classes = 3
    # meta_path
    # item_buy_user_item = scipy.sparse.load_npz(prefix + '/item_buy_user_item.npz').toarray()
    # item_cart_user_item = scipy.sparse.load_npz(prefix + '/item_cart_user_item.npz').toarray()
    # item_pav_user_item = scipy.sparse.load_npz(prefix + '/item_fav_user_item.npz').toarray()
    # item_pv_user_item = scipy.sparse.load_npz(prefix + '/item_pv_user_item.npz').toarray()

    item_buy_user_item = scipy.sparse.load_npz(prefix + '/item_buy_user_item.npz')
    item_cart_user_item = scipy.sparse.load_npz(prefix + '/item_cart_user_item.npz')
    item_pav_user_item = scipy.sparse.load_npz(prefix + '/item_fav_user_item.npz')
    item_pv_user_item = scipy.sparse.load_npz(prefix + '/item_pv_user_item.npz')
    item_item_brand = scipy.sparse.load_npz(prefix + '/item_item_brand.npz')

    item_item_shop = scipy.sparse.load_npz(prefix + '/item_item_shop.npz')

    g1 = dgl.DGLGraph(item_buy_user_item)
    g2 = dgl.DGLGraph(item_cart_user_item)
    g3 = dgl.DGLGraph(item_pav_user_item)
    g4 = dgl.DGLGraph(item_pv_user_item)
    g5 = dgl.DGLGraph(item_item_brand)
    g6 = dgl.DGLGraph(item_item_shop)
    g1 = dgl.add_self_loop(g1)
    g2 = dgl.add_self_loop(g2)
    g3 = dgl.add_self_loop(g3)
    g4 = dgl.add_self_loop(g4)
    g5 = dgl.add_self_loop(g5)
    g6 = dgl.add_self_loop(g6)
    g = [g1, g2, g3, g4]
    # g = [g5,g6]
    NS=[]

    return  g, features, NS,similarity_matrix, labels, num_classes, train_idx, val_idx, test_idx

def load_data6(prefix=r'D:\阿里复现数据集\六分类'):
    features_0 = scipy.sparse.load_npz(prefix + '/product_feature.npz').toarray()  # features_0为商品特征 features_1为用户特征
    features_0 = torch.FloatTensor(features_0)
    features = [features_0]

    similarity_matrix = np.load(prefix + "/similarity.npy")
    similarity_matrix = torch.from_numpy(similarity_matrix).type(torch.FloatTensor)
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    num_classes = 6
    # meta_path
    # item_buy_user_item = scipy.sparse.load_npz(prefix + '/item_buy_user_item.npz').toarray()
    # item_cart_user_item = scipy.sparse.load_npz(prefix + '/item_cart_user_item.npz').toarray()
    # item_pav_user_item = scipy.sparse.load_npz(prefix + '/item_fav_user_item.npz').toarray()
    # item_pv_user_item = scipy.sparse.load_npz(prefix + '/item_pv_user_item.npz').toarray()

    item_buy_user_item = scipy.sparse.load_npz(prefix + '/item_buy_user_item.npz')
    item_cart_user_item = scipy.sparse.load_npz(prefix + '/item_cart_user_item.npz')
    item_pav_user_item = scipy.sparse.load_npz(prefix + '/item_fav_user_item.npz')
    item_pv_user_item = scipy.sparse.load_npz(prefix + '/item_pv_user_item.npz')
    item_item_brand = scipy.sparse.load_npz(prefix + '/item_item_brand.npz')

    item_item_shop = scipy.sparse.load_npz(prefix + '/item_item_shop.npz')

    g1 = dgl.DGLGraph(item_buy_user_item)
    g2 = dgl.DGLGraph(item_cart_user_item)
    g3 = dgl.DGLGraph(item_pav_user_item)
    g4 = dgl.DGLGraph(item_pv_user_item)
    g5 = dgl.DGLGraph(item_item_brand)
    g6 = dgl.DGLGraph(item_item_shop)
    g1 = dgl.add_self_loop(g1)
    g2 = dgl.add_self_loop(g2)
    g3 = dgl.add_self_loop(g3)
    g4 = dgl.add_self_loop(g4)
    g5 = dgl.add_self_loop(g5)
    g6 = dgl.add_self_loop(g6)
    # g=[g1,g2,g3,g4]
    g = [g5,g6]
    # g = [g1,g2,g3,g4,g5,g6]
    NS=[]

    return  g, features, NS,similarity_matrix, labels, num_classes, train_idx, val_idx, test_idx


if __name__ == "__main__":
    load_data()