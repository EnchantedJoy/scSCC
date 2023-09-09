from loss import *
from utils import *
from network import *
import numpy as np
import torch
import os
import h5py
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def saveEmbd(X, Y, Y_pred, epath, dataset):
    path = os.path.join(os.getcwd(), epath, dataset)
    if not os.path.exists(path):
        os.mkdir(path)
    with h5py.File(path + "/embd.h5", 'w') as fw:
        fw.create_dataset("Z", data=X, compression="gzip")
        fw.create_dataset("Y", data=Y, compression="gzip")
        fw.create_dataset("YPre", data=Y_pred, compression="gzip")


def cluster_acc(y_true, y_pred):
    '''
    args:
        y_true: [ndarray]
        y_pred: [ndarray]

    returns:
        acc: float
    '''
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    ## Hungarian algorithm
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros([D, D], dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    # print(w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size


def train(all_data,
          n_classes,
          lab_full=None,
          evaluate_training=True,
          args=None,
          protein_data=None):

    kappa = args.kappa
    alpha = args.alpha
    dropoutRate = args.dropoutRate

    optimizer = args.optim
    batch_size = args.batch_size
    lr = args.lr
    z_dim = args.dim
    enc_dim = args.encoderDim
    headDim = args.headDim
    protein_dim = None
    if protein_data is not None:
        protein_dim = args.protein_dim
    epochs = args.epochs
    preEpochs = args.preEpochs
    activation = args.activation
    threshold = args.threshold

    swav_temperature = args.swav_temperature
    instance_temperature = args.instance_temperature
    epsilon = args.epsilon
    sinkhorn_iterations = args.sinkhorn_iterations

    datasetName = args.dataset
    epath = args.embd_path

    # define autoencoder structure
    input_dim = all_data.shape[1]
    lab_idx = None
    if args.dataset.split("_")[-1] == "MM":
        lab_idx = args.y_idx

    model = swavContrastive(input_dim,
                            z_dim,
                            headDim,
                            n_classes,
                            alpha,
                            activation,
                            dropoutRate,
                            swav_temperature,
                            enc_dim=enc_dim,
                            proteinDim=protein_dim)
    model.to(device)

    # select optimizer, default to be adam
    if optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=lr)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=0.9,
                              nesterov=True)
    elif optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    else:
        raise Exception("Enter correct optimizer name!")

    criterion_swav = swavLoss(swav_temperature)
    criterion_instance = InstanceLoss(instance_temperature, device)

    max_value, best_model = -1, -1
    print("----- Traing phase -----")

    ### ! ***** train with contrastive *******
    idx = np.arange(len(all_data))

    for epoch in range(epochs):
        # print("epoch", epoch)
        adjust_learning_rate(optimizer, epoch, lr)
        np.random.shuffle(idx)
        totalLoss, totalInstanceLoss, totalSwavLoss = 0, 0, 0
        model.train()
        for pre_index in range(len(all_data) // batch_size + 1):
            optimizer.zero_grad()
            with torch.no_grad():
                ## normalize prototypes
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.prototypes.weight.copy_(w)

            c_idx = np.arange(pre_index * batch_size,
                              min(len(all_data), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue

            c_idx = idx[c_idx]
            c_inp = all_data[c_idx]
            c_pro = None
            if protein_data is not None:
                c_pro = torch.FloatTensor(protein_data[c_idx]).to(device)
            input1 = torch.FloatTensor(c_inp).to(device)
            inst1, p_1 = model(input1, c_pro)

            input2 = torch.FloatTensor(c_inp).to(device)
            inst2, p_2 = model(input2, c_pro)

            feature_instance = torch.cat(
                [inst1.unsqueeze(1), inst2.unsqueeze(1)], dim=1)

            with torch.no_grad():
                out_1 = p_1.detach()
                out_2 = p_2.detach()
                q_1 = distributed_sinkhorn(out_1, epsilon, sinkhorn_iterations)
                q_2 = distributed_sinkhorn(out_2, epsilon, sinkhorn_iterations)

            swavloss = criterion_swav(q_1, p_2) + criterion_swav(q_2, p_1)

            if epoch < preEpochs:
                instanceloss = criterion_instance(feature_instance)
                totalInstanceLoss += instanceloss.item()
                batchLoss = swavloss + kappa * instanceloss
            else:
                batchLoss = swavloss

            totalLoss += batchLoss.item()
            totalSwavLoss += swavloss.item()

            batchLoss.backward()
            optimizer.step()

        if evaluate_training and lab_full is not None and epoch >= preEpochs:
            model.eval()
            print(f"epoch: {epoch}")
            with torch.no_grad():
                features = model.encoder(
                    torch.FloatTensor(all_data).to(device))
                lab_pred = model.get_cluster(
                    torch.FloatTensor(all_data).to(device))                
                # if protein_data is None:
                #     features = model.encoder(
                #         torch.FloatTensor(all_data).to(device))
                #     lab_pred = model.get_cluster(
                #         torch.FloatTensor(all_data).to(device))
                # else:
                #     features = model.encoder(
                #         torch.FloatTensor(all_data).to(device),
                #         torch.FloatTensor(protein_data).to(device))
                #     lab_pred = model.get_cluster(
                #         torch.FloatTensor(all_data).to(device),
                #         torch.FloatTensor(protein_data).to(device))
                features = features.detach().cpu().numpy()
                lab_pred = lab_pred.detach().cpu().numpy()
                if lab_idx is not None:
                    features = features[lab_idx]
                    lab_pred = lab_pred[lab_idx]

            res = cluster_embedding(features,
                                    n_classes,
                                    lab_full,
                                    save_pred=True)
            stopACC = cluster_acc(lab_pred, res["pred"])
            stopCondition = cluster_embedding(features,
                                              n_classes,
                                              lab_pred,
                                              save_pred=True)
            stopARI = stopCondition["ari"]
            stopNMI = stopCondition["nmi"]
            pred_ari = round(adjusted_rand_score(lab_full, lab_pred), 4)
            ## add stop condition
            print(
                f"Epoch {epoch}: \n \tLoss: {totalLoss}\n\tinstanceLoss: {totalInstanceLoss}\n\tARI: {res['ari']}\n\tNMI: {res['nmi']}"
            )
            print(f"predict ari: {pred_ari}")
            print(
                f"stopACC: {stopACC}\n stopARI: {stopARI}\n stopNMI: {stopNMI}")

            if stopARI >= max_value:
                max_value = stopARI
                save_model(datasetName, model, optimizer, epoch, best_model)
                saveEmbd(features, lab_full, lab_pred, epath, datasetName)
                best_model = epoch
            if max_value > threshold:
                break

    ## extract model
    model.eval()
    model_fp = os.getcwd(
    ) + '/' + args.save_path + '/' + datasetName + "/checkpoint_{}.tar".format(
        best_model)
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)
    with torch.no_grad():
        if protein_data is not None:
            features = model.encoder(
                torch.FloatTensor(all_data).to(device),
                torch.FloatTensor(protein_data).to(device))
        else:
            features = model.encoder(
                torch.FloatTensor(all_data).to(device))            
        features = features.detach().cpu().numpy()

    return features, best_model


@torch.no_grad()
def distributed_sinkhorn(out, epsilon, iters):
    Q = torch.exp(out / epsilon).t(
    )  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()
