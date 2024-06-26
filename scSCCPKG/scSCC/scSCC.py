from .loss import *
from .utils import *
from .network import *
import numpy as np
import torch
import os
import h5py
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class scSCC():
    def __init__(self,
                 input_data,
                 n_clusters,
                 dataset,
                 seed=0,
                 kappa=-1,
                 alpha=0.1,
                 insT=0.1,
                 swpT=0.1,
                 epochs=200,
                 preEpochs=150,
                 sinkEps=0.05,
                 sinkIter=3,
                 batchSize=256,
                 learningRate=0.01,
                 encoderDims=[200, 40],
                 z_dim=60,
                 head_dim=40,
                 dropout=0.9,
                 threshHold=0.94,
                 device='cpu') -> None:
        ######### set up random seed
        setup_seed(seed)

        #########   some parameters ##########
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.dataset = dataset
        self.n_cells, self.input_dim = input_data.shape
        if kappa != -1:
            self.kappa = kappa
        else:
            self.kappa = 0.01 if self.n_cells < 10000 else 1.
        self.insT, self.swpT = insT, swpT
        self.sinkEps, self.sinkIter = sinkEps, sinkIter
        self.encoderDims = encoderDims
        self.z_dim, self.head_dim = z_dim, head_dim
        self.learningRate = learningRate
        self.epochs, self.preEpochs = epochs, preEpochs
        self.batchSize = batchSize

        self.dropoutRate = dropout
        self.threshHold = threshHold
        self.activation = 'relu'
        self.device = torch.device(device)

        self.input_data = torch.FloatTensor(input_data)
        self.model = scSCCNet(self.input_dim,
                              self.z_dim,
                              self.head_dim,
                              self.n_clusters,
                              self.alpha,
                              self.activation,
                              self.dropoutRate,
                              self.swpT,
                              enc_dim=self.encoderDims).to(self.device)
        ### ============= criterions ============
        self.criterion_swav = swavLoss(self.swpT)
        self.criterion_instance = InstanceLoss(self.insT, device)

    def train(self):
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.model.parameters()),
                                          lr=self.learningRate)
        max_value, best_model = -1, -1
        print("----- Traing phase -----")

        ###  ***** train with contrastive *******
        idx = np.arange(self.n_cells)

        for epoch in tqdm(range(self.epochs)):
            # print("epoch", epoch)
            adjust_learning_rate(self.optimizer, epoch, self.learningRate)
            np.random.shuffle(idx)
            totalLoss, totalInstanceLoss, totalSwavLoss = 0, 0, 0
            self.model.train()
            for pre_index in range(self.n_cells // self.batchSize + 1):
                self.optimizer.zero_grad()
                with torch.no_grad():
                    ## normalize prototypes
                    w = self.model.prototypes.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    self.model.prototypes.weight.copy_(w)

                c_idx = np.arange(
                    pre_index * self.batchSize,
                    min(self.n_cells, (pre_index + 1) * self.batchSize))
                if len(c_idx) == 0:
                    continue

                c_idx = idx[c_idx]
                c_inp = self.input_data[c_idx]
                input1 = torch.FloatTensor(c_inp).to(self.device)
                inst1, p_1 = self.model(input1)

                input2 = torch.FloatTensor(c_inp).to(self.device)
                inst2, p_2 = self.model(input2)

                feature_instance = torch.cat(
                    [inst1.unsqueeze(1),
                     inst2.unsqueeze(1)], dim=1)

                with torch.no_grad():
                    out_1 = p_1.detach()
                    out_2 = p_2.detach()
                    q_1 = distributed_sinkhorn(out_1, self.sinkEps, self.sinkIter)
                    q_2 = distributed_sinkhorn(out_2, self.sinkEps, self.sinkIter)

                swavloss = self.criterion_swav(q_1, p_2) + self.criterion_swav(
                    q_2, p_1)

                if epoch < self.preEpochs:
                    instanceloss = self.criterion_instance(feature_instance)
                    totalInstanceLoss += instanceloss.item()
                    batchLoss = swavloss + self.kappa * instanceloss
                else:
                    batchLoss = swavloss

                totalLoss += batchLoss.item()
                totalSwavLoss += swavloss.item()

                batchLoss.backward()
                self.optimizer.step()

            if epoch >= self.preEpochs:
                self.model.eval()
                print(f"epoch: {epoch}")
                with torch.no_grad():
                    features = self.model.encoder(
                        torch.FloatTensor(self.input_data).to(self.device))
                    lab_pred = self.model.get_cluster(
                        torch.FloatTensor(self.input_data).to(self.device))
                    features = features.detach().cpu().numpy()
                    lab_pred = lab_pred.detach().cpu().numpy()

                res = cluster_embedding(features,
                                        self.n_clusters,
                                        save_pred=True)
                stopCondition = cluster_embedding(features,
                                                  self.n_clusters,
                                                  lab_pred,
                                                  save_pred=True)
                stopARI = stopCondition["ari"]
                stopNMI = stopCondition["nmi"]
                ## add stop condition
                print(
                    f"Epoch {epoch}: \n \tLoss: {totalLoss}\n\tinstanceLoss: {totalInstanceLoss}"
                )
                print(f"\tstopARI: {stopARI}\t stopNMI: {stopNMI}")

                if stopARI >= max_value:
                    max_value = stopARI
                    save_embed(features, self.dataset, epoch)
                    save_model(self.dataset, self.model, self.optimizer, epoch,
                               best_model)
                    best_model = epoch
                if max_value > self.threshHold:
                    break

        return np.load(f"./embd/{self.dataset}_{best_model}.npy")

        ## extract model
        # self.model.eval()
        # model_fp = os.getcwd(
        # ) + '/' + 'save' + '/' + self.dataset + "/checkpoint_{}.tar".format(
        #     best_model)
        # self.model.load_state_dict(
        #     torch.load(model_fp, map_location=self.device.type)['net'])
        # self.model.to(self.device)
        # with torch.no_grad():
        #     features = self.model.encoder(torch.FloatTensor(self.input_data).to(self.device))
        #     features = features.detach().cpu().numpy()
        #     if not os.path.exists("./scSCCFeatures"):
        #         os.mkdir('./scSCCFeatures')
        #     np.save(f"./scSCCFeatures/scSCC_{self.dataset}.npy", features)
        # return features



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
