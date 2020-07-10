''' main module '''
import torch
import torchvision.datasets
import VGG
import trainVGG
import graphics

CIFAR_train = torchvision.datasets.CIFAR10('./', download=True, train=True)
CIFAR_test = torchvision.datasets.CIFAR10('./', download=True, train=False)

X_train = torch.FloatTensor(CIFAR_train.data)
y_train = torch.LongTensor(CIFAR_train.targets)
X_test = torch.FloatTensor(CIFAR_test.data)
y_test = torch.LongTensor(CIFAR_test.targets)

# normilize ant permutate data to suit torch
X_train /= 255.
X_test /= 255.
X_train = X_train.permute(0, 3, 1, 2)
X_test = X_test.permute(0, 3, 1, 2)
print(f'X_train shape: {X_train.shape}',
    f'X_test shape: {X_test.shape}',
    f'y_train shape: {y_train.shape}',
    f'y_test shape: {y_test.shape}'
)

def main():
    net_list = []
    accuracies = {}
    losses = {}
    accuracies['VGG11'], losses['VGG11'] = trainVGG.train2(VGG.VGG(), X_train, y_train, X_test, y_test)
    net_list.append(VGG.VGG())
    graphics.acc_loss_graph(accuracies, losses, net_list)
main()