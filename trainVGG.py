'''train VGG'''
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter 

def adjust_learning_rate(optimizer, lr, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    if epoch%50 == 0:
        lr = lr / 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def save_checkpoint(state, filename="my_model.pth.tar"):
    print('==> saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, net, optimizer, load_optimizer=False):
    print("==> Loading checkpoint")
    net.load_state_dict(checkpoint['state_dict'])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

writer = SummaryWriter(f'runs/CIFAR/trying_tb') # specify files which tensorboard can read

def train(net, X_train, y_train, X_test, y_test, optimizer, lr, loss, batch_size=256, epoch_num=60, epoch_info_show=10,\
        save_net_state=False, load_model=False, verbose=True):
    if load_model:
        load_checkpoint(torch.load("my_model.pth.tar"), net, optimizer)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = loss.to(device)
    net = net.to(device)
    t = time.time()
    test_accuracy_history = []
    test_loss_history = []
    train_accuracy_history = []
    train_loss_history = []
    
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    step = 0
    for epoch in range(1, epoch_num+1):
        train_accuracy_history_batch = []
        train_loss_history_batch = []
        # adjust_learning_rate(optimizer,lr,epoch)    
        if save_net_state and epoch%10 == 0:
            checkpoint = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint, 'my_model19.pth.tar')

        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            net.train()

            batch_indexes = order[start_index:start_index+batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = net.forward(X_batch)
            loss_value = loss(preds, y_batch)

            train_loss_history_batch.append(loss_value)
            accuracy_train = (preds.argmax(dim=1) == y_batch).float().mean().item()
            train_accuracy_history_batch.append(accuracy_train)

            loss_value.backward()

            optimizer.step()
            writer.add_scalar('Training loss', loss_value, global_step=step )
            writer.add_scalar('Training accuracy',  accuracy_train, global_step=step)
            step += 1
        print(f'train loss at current epoch: {sum(train_loss_history_batch)/len(train_loss_history_batch)}')
        print(f'train accuracy at current epoch: {sum(train_accuracy_history_batch)/len(train_accuracy_history_batch)}')
        train_loss_history.append(sum(train_loss_history_batch)/len(train_loss_history_batch))
        train_accuracy_history.append(sum(train_accuracy_history_batch)/len(train_accuracy_history_batch))

        net.eval()
        with torch.no_grad():
            test_preds = net.forward(X_test)
            loss_value = loss(test_preds, y_test).item()
            test_loss_history.append(loss_value)

            accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().item()
            test_accuracy_history.append(accuracy)
        
            if verbose:
                print('Train Epoch: {} Time: {} Accuracy: {}, Loss: {}, GPU_Mem_alloc: {} GPU_Mem_cashed: {}'\
                  .format(epoch, time.strftime("%H:%M:%S", time.gmtime(time.time() - t)), accuracy, loss_value, \
                            torch.cuda.memory_allocated(), torch.cuda.memory_cached()))
              

    return test_accuracy_history, test_loss_history