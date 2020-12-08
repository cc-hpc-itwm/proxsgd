import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import  compute_cdf, print_nonzeros
from IPython import display


def train(model,train_data,weight_reg,logger,optimizer,epoch,retrain,summary):

    if retrain== True:
        print('\nIn Retraining...')
    else:
        print('\nIn Training...')
    model.train()
    lossvals = []
    l1loss = []
    criterion = nn.CrossEntropyLoss()
    num_batches = len(train_data)
    for batch_id, (data, target) in enumerate(train_data):
        data, target = Variable(data.cuda()), Variable(target.cuda())

        out = model(data)
        loss = criterion(out, target)
        #Save the loss value after each iteration into this array
        lossvals.append(loss.item())

        #Accumulate the L1 loss across all layers and save it into the l1loss array
        laccum = 0
        for name, param in model.named_parameters():
            if 'bias' not in name:
                l1 = torch.sum(torch.abs(param))
                if weight_reg is not None:
                    loss = loss + (weight_reg * l1)
                laccum += l1.item()
        l1loss.append(laccum)

        optimizer.zero_grad()
        loss.backward()
        if retrain ==  True:
            for index,module in enumerate(model.modules()):
                if isinstance(module, nn.Conv2d):
                    weight_copy = module.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().cuda()
                    module.weight.grad.data.mul_(mask)
        optimizer.step()
        step = epoch * num_batches + batch_id
        
                
        if batch_id % 20 == 0:
            summary.add_scalar("Training Loss", loss.item(), step)
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_id * len(data), len(train_data.dataset),
                100. * batch_id / len(train_data), loss.item()))
            
    return lossvals, l1loss

def test(model,test_loader,logger,summary,epoch):
    model.eval()
    test_loss = 0
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    num_batches = len(test_loader)
    with torch.no_grad():
        for batch_id,(data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss=criterion(output, target)
            test_loss += loss
            _, pred = output.topk(5, 1, largest=True, sorted=True)
            target = target.view(target.size(0), -1).expand_as(pred)
            correct = pred.eq(target).float()
            
            
            correct_5 += correct[:, :5].sum()
            #compute top1
            correct_1 += correct[:, :1].sum()
            step = epoch * num_batches + batch_id
            if batch_id % 20 == 0:
                if summary!=None:
                    summary.add_scalar("Test Loss", loss.item(), step)
                

    test_loss /= len(test_loader.dataset)
    top1 = 100. * correct_1 / len(test_loader.dataset)
    top5 = 100. * correct_5 / len(test_loader.dataset)

    logger.info("Top 1 accuracy: {:.2f}%".format(top1))
    logger.info("Top 5 accuracy: {:.2f}%".format(top5))
    compression_rate=print_nonzeros(model)
    #logger.info("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    return test_loss, top1.cpu().numpy(), top5.cpu().numpy(),compression_rate

def evaluate(model,test_loader,logger):
    loss, acc1, acc5 = test(model,test_loader, logger)
    #print_nonzeros(model)
    sparsity_x, sparsity_y = compute_cdf(model)
    plt.plot(sparsity_x,sparsity_y)
    return loss, acc1, acc5 