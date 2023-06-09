import torch.nn as nn
import torch as th
import torch.nn.functional as F
import torch


class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j


class NormSoftmaxLoss2(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)
        return - loss_i


class NCE(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        targets = torch.arange(0, x.size(0),
                               dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits / self.temperature, targets,
                               reduction='mean')
        return loss


class MMS_Loss(nn.Module):
    def __init__(self, margin=0.001):
        super(MMS_Loss, self).__init__()
        self.margin = margin

    def forward(self, S, ):
        deltas = self.margin * torch.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=1, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]

        x1 = th.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = th.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = th.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = th.ones(x.shape) - th.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = th.nonzero(th.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = th.index_select(x1, dim=0, index=keep_idx)
            x2_ = th.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def nll_loss(output, target):
    return F.nll_loss(output, target)


if __name__ == "__main__":
    import torch

    random_sims = (torch.rand([10, 8]) * 2) - 1
    loss = NormSoftmaxLoss()
    loss(random_sims)


def normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm

def normalize_embeddings_seq(a, eps=1e-8):
    # input: batch by seq by feat
    a_n = a.norm(dim=2)[:, :, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm

def sim_matrix(a, b, eps=1e-8, flag_normalize_embeddings=True):
    """
    added eps for numerical stability
    """
    if flag_normalize_embeddings:
        a = normalize_embeddings(a, eps)
        b = normalize_embeddings(b, eps)

    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt
