import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CrossEntropyReduction(nn.Module):
    def __init__(self, betta=0.001, dim=-1):
        super().__init__()
        self.betta = betta
        self.dim = dim
    def forward(self, output, target):
        logsoftmax = nn.LogSoftmax(dim=self.dim)
        probabilities = F.softmax(output, dim=self.dim)
        entropy = torch.sum(-probabilities*logsoftmax(output))
        loss = F.relu(F.cross_entropy(output, target) - self.betta*entropy)
        return loss

if __name__ == '__main__':
    loss = CrossEntropyReduction(0.01)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print(target)
    output = loss(input, target)
    output.backward()
    print(output)