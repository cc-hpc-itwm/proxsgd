import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import torch
import torch.nn as nn
import config
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

from dataset import load_cifar100, load_mnist
from models.densenet import densenet201

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

def plot_learning_curve(model,data, xlabel, ylabel, filename, ylim=None, cdf_data=False):
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
        plt.yticks(np.arange(ylim[0], ylim[1],10))
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
    plt.close('all')

def print_nonzeros(model):
    nonzero = total = 0
    for name,param in model.named_parameters():
        if "weight" in name:
            nz_count=param.nonzero().size(0)
            total_params = param.numel()
            nonzero += nz_count
            total += total_params
            #print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'Active: {nonzero}, Pruned : {total - nonzero}, Total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    compression_rate= round(total/nonzero,2)
    percentage_pruned= round(100*(total-nonzero) / total,2)
    return str(compression_rate)+'x'+'('+str(percentage_pruned)+'% pruned)'

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, is_best, current_loss, current_acc1, current_acc5, train_dir, original_model_acc1, pruned_model_acc1, pruned_threshold, run_id ):
    """
    Saves model checkpoint.
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param model: model
    :param optimizer: optimizer to update weights
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
            'epochs_since_improvement': epochs_since_improvement,
            'model':model,
            'optimizer': optimizer,
            'error' : current_loss,
            'current_acc1': current_acc1, 
            'current_acc5':current_acc5,
            'original_model_acc1': original_model_acc1,
            'pruned_model_acc1': pruned_model_acc1, 
             'pruned_threshold': pruned_threshold,
             'run_id' : run_id
            }
    
    torch.save(state, train_dir+config.filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, train_dir+'BEST_' + config.filename)

if __name__ == "__main__":
    quality_parameter = 2.1 
    model_path="experiment_densenet_CIFAR100_2.1/densenet_weight_retrained_2.1"
    model = densenet201()
    model = model.cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    train_data, test_loader = load_cifar100()
    lossarr=np.load('experiment_densenet_CIFAR100_2.1/densenet_loss_2.1.npy')
    print(lossarr)
    plot_learning_curve(model,data=lossarr, xlabel="Iteration", ylabel="Training Loss", filename="experiment_densenet_CIFAR100_2.1/densenet_loss1_"+str(quality_parameter)+".png")
    
