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

LOSS = torch.nn.CrossEntropyLoss()
NET = VGG.VGG('E')
OPTIMIZER = torch.optim.SGD(NET.parameters(), momentum=0.9, lr=0.01, weight_decay=0.001)

def main():
    net_list = []
    accuracies = {}
    losses = {}
    accuracies[NET], losses[NET] = trainVGG.train(NET, X_train, y_train, X_test, y_test, OPTIMIZER, LOSS, save_net_state=True, 
                                                            load_model=False)
    net_list.append(NET)
    graphics.acc_loss_graph(accuracies, losses, net_list, download=True)
main()
    