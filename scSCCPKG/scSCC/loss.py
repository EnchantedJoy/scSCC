import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class swavLoss(nn.Module):
    def __init__(self, temperature) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, q, p):
        p = p / self.temperature
        return -torch.mean(torch.sum(q * F.log_softmax(p, dim=1), dim=1))


class InstanceLoss(nn.Module):
    def __init__(self, temperature, device):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, features):

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        self.batch_size = features.shape[0]
        self.mask = self.mask_correlated_samples(self.batch_size).to(
            self.device)
        N = 2 * self.batch_size
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        sim = torch.matmul(contrast_feature,
                           contrast_feature.T) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N

        return loss