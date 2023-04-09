import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 class_weights,
                 ignore_index=255,
                 use_top_k=False,
                 top_k_ratio=1.0,
                 future_discount=1.0,
                 use_focal = True,
                 gamma = 2.0,
                 alpha = 1.0
                 ):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount
        self.use_focal = use_focal

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, prediction, target):
        b, c, h, w = prediction.shape

        if self.use_focal:
            eps: float = 1e-8
            input_soft: torch.Tensor = F.softmax(prediction, dim=1) + eps

            # create the labels one hot tensor
            one_hot = torch.zeros_like(prediction,dtype=target.dtype).cuda()

            target_one_hot = one_hot.scatter_(1, target.unsqueeze(1), 1.0) + eps

            # print(target_one_hot.shape)

            weight = torch.pow(-input_soft + 1., self.gamma)
            weight_set = self.class_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,weight.shape[2],weight.shape[3])
            focal = -self.alpha * weight * torch.log(input_soft) * weight_set.to(weight.device)
            loss = torch.sum(target_one_hot * focal.to(target_one_hot.device) , dim=1)
            loss = loss.view(b,-1)

        else:
            loss = F.cross_entropy(
                prediction,
                target,
                ignore_index=self.ignore_index,
                reduction='none',
                weight=self.class_weights.cuda(),
            )

            loss = loss.view(b, h, w)
            loss = loss.view(b, -1)

            if self.use_top_k:
                # Penalises the top-k hardest pixels
                k = int(self.top_k_ratio * loss.shape[1])
                loss, _ = torch.sort(loss, dim=1, descending=True)
                loss = loss[:, :k]

        return torch.mean(loss)
