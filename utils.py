import numpy as np
import os
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import time
import math
from scipy.spatial.distance import cdist


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def adjust_learning_rate(optimizer, epoch, lr):
    p = {
        'epochs': 500,
        'optimizer': 'sgd',
        'optimizer_kwargs':
            {'nesterov': False,
             'weight_decay': 0.0001,
             'momentum': 0.9,
             },
        'scheduler': 'cosine',
        'scheduler_kwargs': {'lr_decay_rate': 0.1},
    }

    new_lr = None

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            new_lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        new_lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return lr


def save_model(name, model, optimizer, current_epoch, pre_epoch):
    if pre_epoch != -1:
        pre_path = os.path.join(os.getcwd(), "save", name, "checkpoint_{}.tar".format(pre_epoch))
        os.remove(pre_path)
    cur_path = os.path.join(os.getcwd(), "save", name, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, cur_path)



def mixupCells(data, n_neighbors=5):
    print("create adjacent matrix from pca expr --------------->")
    distance_mat = cdist(data, data, metric="correlation")  ## + res is distance corr = 1 - dist
    distance2weight = 1 - distance_mat  ## translate dist to pearson correlation
    threshold = np.sort(distance2weight)[:, -n_neighbors - 1:-n_neighbors]
    distance2weight[distance2weight < threshold] = 0 
    distance2weight = (distance2weight + distance2weight.T) / 2
    print("knn graph created ----<")
    return distance2weight


def cluster_embedding(embedding, cluster_number, real_label, save_pred=False, cluster_methods=None):
    if cluster_methods is None:
        cluster_methods = ["KMeans"]
    result = {"t_clust": time.time()}
    if "KMeans" in cluster_methods:
        kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=0)
        pred = kmeans.fit_predict(embedding)
        result[f"sc_pred"] = round(silhouette_score(embedding, pred), 4)
        result[f"db_pred"] = round(davies_bouldin_score(embedding, pred), 4)
        if real_label is not None:
            result[f"ari"] = round(adjusted_rand_score(real_label, pred), 4)
            result[f"nmi"] = round(normalized_mutual_info_score(real_label, pred), 4)
            result[f"sc_real"] = round(silhouette_score(embedding, real_label), 4)
            result[f"db_real"] = round(davies_bouldin_score(embedding, real_label), 4)
        result["t_k"] = time.time()
        if save_pred:
            result[f"pred"] = pred

    return result


## - add in 2023_4_19
## - add knn augmentation



def empty_safe(fn, dtype):

    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)

    return _fn

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
