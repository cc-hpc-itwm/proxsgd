import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from ProxSGD_optimizer import ProxSGD
import matplotlib.pyplot as plt
from models.densenet import densenet201


def start_dense201(epsilon, epsilon_decay, rho, rho_decay, mu, gamma, clip_bounds):

    def get_param_vec(model):
        #flatten parameters of the network into a numpy array
        rv = torch.zeros(1,0).cuda()
        for p in model.parameters():
            rv_tmp = p.data.view(1,-1)
            rv = torch.cat((rv, rv_tmp),1)
        return rv.view(-1).cpu().numpy()

    def plot_sparsity(model,filename):
        #generate a sparsity plot of a given network
        Num_pars = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data = get_param_vec(model)
        values, base = np.histogram(data, bins=1000000)
        cumulative = np.cumsum(values)
        f = plt.figure()
        plt.plot(base[:-1], cumulative/Num_pars, c='blue')
        plt.grid(True)
        plt.xlabel("Weights value")
        plt.ylabel("CDF")
        plt.savefig(filename)
        plt.close()


    def save_sparsity(model):
        #save sparsity values
        Num_pars = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data = get_param_vec(model)
        values, base = np.histogram(data, bins=1000000)
        cumulative = np.cumsum(values)
        return base[:-1],cumulative/Num_pars

    #Load the dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = torch.utils.data.DataLoader(datasets.CIFAR100(root='dataset', train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize,]), download=True),batch_size=128,num_workers=6, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR100(root='dataset', train=False, transform=transforms.Compose([transforms.ToTensor(),normalize,])),batch_size=128, num_workers=4, shuffle=False)

    #load the model and place it onto the GPU
    model = densenet201()
    model = model.cuda()

    #Initialize the optimizer with the given parameters
    optimizer = ProxSGD(model.parameters(), epsilon=epsilon, epsilon_decay=epsilon_decay, rho=rho, rho_decay=rho_decay, mu=mu, gamma=gamma, clip_bounds=clip_bounds)

    #Remove weight regulatization in Pytorch as ProxSGD already has it implemented
    weight_reg = None

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
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_data.dataset),
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
        print("Top 1 err: {:.0f}%".format( (1 - correct_1 / len(test_loader.dataset))*100))
        print("Top 5 err: {:.0f}%".format( (1 - correct_5 / len(test_loader.dataset))*100))
        print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

        top1 = 100. * correct_1 / len(test_loader.dataset)
        top5 = 100. * correct_5 / len(test_loader.dataset)

        return top1.cpu().numpy(), top5.cpu().numpy()

    testarr1 = []
    testarr5 = []
    lossarr = []
    l1_loss = []
    sparsity_x = []
    sparsity_y = []

    torch.save(model.state_dict(),"Densenet_weight_iter0")
    #do an initial accuracy run
    acc1, acc5 = test()
    testarr1.append(acc1)
    testarr5.append(acc5)

    for ep in range(60):
        #train network
        loss, l1it = train(ep)
        lossarr.append(loss)
        l1_loss.append(l1it)
        #test network and save testerrors
        acc1, acc5 = test()
        testarr1.append(acc1)
        testarr5.append(acc5)
        #calculate the sparsity of the network
        sparsity1, sparsity2 = save_sparsity(model)
        sparsity_x.append(sparsity1)
        sparsity_y.append(sparsity2)

    #save the network weights
    torch.save(model.state_dict(),"Densenet_weight_final")

    plot_sparsity(model,"plot_Densenet_sparsity_"+str(ep)+".png")

    #save all the results
    lossarr = np.hstack(lossarr)
    np.save("Densenet_loss_cifar100",lossarr)
    np.save("Densenet_l1loss_cifar100",l1_loss)
    np.save("Densenet_acc1_cifar100",testarr1)
    np.save("Densenet_acc5_cifar100",testarr5)
    np.save("Densenet_spars_x_cifar100",sparsity_x)
    np.save("Densenet_spars_y_cifar100",sparsity_y)

    #plot resulting loss and accuracies
    f = plt.figure()
    plt.plot(lossarr)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.grid(True)
    plt.savefig("plot_Densenet_loss_cifar100.png")


    f = plt.figure()
    plt.plot(testarr1)
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.grid(True)
    plt.savefig("plot_Densenet_acc1_cifar100.png")

    f = plt.figure()
    plt.plot(testarr5)
    plt.xlabel("Epoch")
    plt.ylabel("Top-5 Accuracy")
    plt.grid(True)
    plt.savefig("plot_Densenet_acc5_cifar100.png")



#Start a run
start_dense201(epsilon=0.21, epsilon_decay=0.6, rho=0.9, rho_decay=0.5, mu=1e-5, gamma=4, clip_bounds=(None,None))
