import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from ProxSGD_optimizer import ProxSGD
from models.densenet import densenet201
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

def train_dense201(epsilon, epsilon_decay, rho, rho_decay, mu, num_epochs=60, run=1):
        
    def set_logger(logger_name, level=logging.INFO):
        """
        Method to return a custom logger with the given name and level
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        log_format = logging.Formatter("%(asctime)s %(message)s", '%Y-%m-%d %H:%M')

        # Creating and adding the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        # Creating and adding the file handler
        file_handler = logging.FileHandler(logger_name, mode='w')
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        return logger

    def get_param_vec(model):
        #flatten parameters of the network into a numpy array
        rv = torch.zeros(1,0).cuda()
        for p in model.parameters():
            rv_tmp = p.data.view(1,-1)
            rv = torch.cat((rv, rv_tmp),1)
        return rv.view(-1).cpu().numpy()

    def compute_cdf(model):
        #save sparsity values
        Num_pars = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data = get_param_vec(model)
        values, base = np.histogram(data, bins=1000000)
        cumulative = np.cumsum(values)
        return base[:-1],cumulative/Num_pars

    def plot_learning_curve(data, xlabel, ylabel, filename, ylim=None, cdf_data=False):
        
        if cdf_data:
            fig, ax = plt.subplots()
            #generate a sparsity plot of a given network
            Num_pars = sum(p.numel() for p in model.parameters() if p.requires_grad)
            data = get_param_vec(model)
            values, base = np.histogram(data, bins=1000000)
            cumulative = np.cumsum(values)
            plt.plot(base[:-1], cumulative/Num_pars)
        else:
            fig = plt.figure()
            plt.plot(data)            
            
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
            
        if cdf_data:
            axins_lower = inset_axes(ax, 1.3,1.3 , loc=2,bbox_to_anchor=(0.6, 0.55),bbox_transform=ax.figure.transFigure) # no zoom
            axins_lower.plot(base[:-1], cumulative/Num_pars)
            axins_lower.set_xlim(-3e-4, 1e-4) # apply the x-limits
            axins_lower.set_ylim(0, 0.2)
            axins_lower.grid(True)
            mark_inset(ax, axins_lower, loc1=2, loc2=4, fc="none", ec="0.5")

            axins_upper = inset_axes(ax, 1.3, 1.3, loc=2, bbox_to_anchor=(0.2, 0.55), bbox_transform=ax.figure.transFigure) # no zoom
            axins_upper.plot(base[:-1], cumulative/Num_pars)
            axins_upper.set_xlim(-1e-4, 3e-4) # apply the x-limits
            axins_upper.set_ylim(0.8, 1)
            axins_upper.grid(True)
            mark_inset(ax, axins_upper, loc1=2, loc2=4, fc="none", ec="0.5")
            
        plt.savefig(filename)

    def train(epoch):
        model.train()
        lossvals = []
        l1loss = []
        criterion = nn.CrossEntropyLoss()

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
            optimizer.step()

            if batch_id % 20 == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_id * len(data), len(train_data.dataset),
                    100. * batch_id / len(train_data), loss.item()))

        return lossvals, l1loss

    def test():
        model.eval()
        test_loss = 0
        correct_1 = 0.0
        correct_5 = 0.0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += criterion(output, target).item()
                _, pred = output.topk(5, 1, largest=True, sorted=True)
                target = target.view(target.size(0), -1).expand_as(pred)
                correct = pred.eq(target).float()

                #compute top 5
                correct_5 += correct[:, :5].sum()

                #compute top1
                correct_1 += correct[:, :1].sum()

        test_loss /= len(test_loader.dataset)
        top1 = 100. * correct_1 / len(test_loader.dataset)
        top5 = 100. * correct_5 / len(test_loader.dataset)

        logger.info("Top 1 accuracy: {:.0f}%".format(top1))
        logger.info("Top 5 accuracy: {:.0f}%".format(top5))
        logger.info("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

        return top1.cpu().numpy(), top5.cpu().numpy()

    #Load the dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = torch.utils.data.DataLoader(datasets.CIFAR100(root='dataset', train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize,]), download=True),batch_size=128,num_workers=6, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR100(root='dataset', train=False, transform=transforms.Compose([transforms.ToTensor(),normalize,])),batch_size=128, num_workers=4, shuffle=False)

    #load the model and place it onto the GPU
    model = densenet201()
    model = model.cuda()

    #Initialize the optimizer with the given parameters
    optimizer = ProxSGD(model.parameters(), epsilon=epsilon, epsilon_decay=epsilon_decay, rho=rho, rho_decay=rho_decay, mu=mu)

    #Remove weight regulatization in Pytorch as ProxSGD already has it implemented
    weight_reg = None
    
    #Initialization: save the initial weights and do an initial accuracy run
    run_id = "epsilon_"+str(epsilon)+"_epsilon_decay_"+str(epsilon_decay)+"_rho_"+str(rho)+"_rho_decay_"+str(rho_decay)+"_mu_"+str(mu)+"_run_"+str(run)    
    torch.save(model.state_dict(),"data/Densenet_weight_iter0_"+run_id)

    logger = set_logger(logger_name="data/logging_"+run_id)
    logger.info("epsilon: "+str(epsilon)+", epsilon_decay: "+str(epsilon_decay)+", rho: "+str(rho)+", rho_decay: "+str(rho_decay)+", mu: "+str(mu)+", run: "+str(run))
    logger.info("Initial accuracies:")
    
    testarr1 = []
    testarr5 = []
    lossarr = []
    l1_loss = []
    weights_cdf_x = []
    weights_cdf_y = []

    acc1, acc5 = test()
    testarr1.append(acc1)
    testarr5.append(acc5)

    #training
    for epoch in range(num_epochs):
        #train network
        loss, l1it = train(epoch)
        lossarr.append(loss)
        l1_loss.append(l1it)
        
        #test network and save testerrors
        acc1, acc5 = test()
        testarr1.append(acc1)
        testarr5.append(acc5)
        
        #compute the sparsity of the network
        sparsity_x, sparsity_y = compute_cdf(model)
        weights_cdf_x.append(sparsity_x)
        weights_cdf_y.append(sparsity_y)

    #training is completed, and save the final weights and learning curves
    torch.save(model.state_dict(),"data/Densenet_weight_final_"+run_id)
    lossarr = np.hstack(lossarr)
    np.save("data/Densenet_cifar100_loss_"+run_id,lossarr)
    np.save("data/Densenet_cifar100_L1___"+run_id,l1_loss)
    np.save("data/Densenet_cifar100_acc1_"+run_id,testarr1)
    np.save("data/Densenet_cifar100_acc5_"+run_id,testarr5)
    np.save("data/Densenet_cifar100_cdfx_"+run_id,weights_cdf_x)
    np.save("data/Densenet_cifar100_cdfy_"+run_id,weights_cdf_y)

    #plot sparsity, training loss, top-1 accuracy and top-5 accuracy
    plot_learning_curve(data=model, xlabel="Weights Value", ylabel="CDF", filename="data/plot_cdf__"+run_id+".png", cdf_data=True)
    plot_learning_curve(data=lossarr, xlabel="Iteration", ylabel="Training Loss", filename="data/plot_loss_"+run_id+".png")
    plot_learning_curve(data=testarr1, xlabel="Epoch", ylabel="Top-1 Accuracy", filename="data/plot_acc1_"+run_id+".png", ylim=[0, 100])
    plot_learning_curve(data=testarr5, xlabel="Epoch", ylabel="Top-5 Accuracy", filename="data/plot_acc5_"+run_id+".png", ylim=[0, 100])

if __name__ == "__main__":
    num_runs = 1 # number of experiments
    start_run = 1
    num_epochs = 60 # number of epochs
    for run in range(start_run, num_runs+1, 1):
        train_dense201(epsilon=0.2122, epsilon_decay=0.5981, rho=0.9, rho_decay=0.5, mu=1e-5, num_epochs=num_epochs, run=run)