# This file is used for hyperparameter search using Random Search. 'config.py' consist all the variables information 
# such as result directory, dataset, training epochs , retraining epochs, network and so on. To run the trials, create a study object,
# which sets the direction of optimization ("maximize" or "minimize"), along with other settings.


import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import math
import optuna
import torch
import torch.nn as nn
import config
import csv
from optuna.samplers import RandomSampler
from os import path, makedirs
from torchvision import datasets, transforms
from torch.autograd import Variable
from prune import prune_model
from trainer import train, test
from ProxSGD_optimizer import ProxSGD
from dataset import load_cifar100, load_mnist
from models.densenet import densenet201
from utils import set_logger, get_param_vec, compute_cdf, plot_learning_curve, print_nonzeros, save_checkpoint
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

from multiprocessing import Manager
import random
import time
from joblib import parallel_backend

def train_dense201(model,optimizer,run_id, result_dir,logger,original_model_acc1, pruned_model_acc1, pruned_threshold,retrain):

    testarr1 = []
    testarr5 = []
    lossarr = []
    l1_loss = []
    weights_cdf_x = []
    weights_cdf_y = []
    epochs_since_improvement = 0
    max_loss = math.inf

    loss, acc1, acc5, compression_rate = test(model, test_loader, logger)
    testarr1.append(acc1)
    testarr5.append(acc5)

    #training/retraining

    if retrain==True:
        num_epochs= config.retrain_epochs
        train_dir= result_dir+'/retrain/'+str(pruned_threshold)+'/'
        print('\nRetraining for Pruning Threshold :', pruned_threshold)
    else:
        num_epochs=config.train_epochs
        train_dir= result_dir+'/train/'
    
    if not path.exists(path.dirname(train_dir)):   # Check if folder/Path for experiment result exists
        makedirs(path.dirname(train_dir))          # If not, then create one   
    

    
    for epoch in range(num_epochs):

        if epochs_since_improvement == 10 :
            break
        #train network
        loss, l1it = train(model,train_data,config.weight_reg,logger,optimizer,epoch,retrain)
        lossarr.append(loss)
        l1_loss.append(l1it)
        
        #test network and save testerrors
        current_loss, acc1, acc5, compression_rate = test(model, test_loader, logger)
        testarr1.append(acc1)
        testarr5.append(acc5)
        
        #compute the sparsity of the network
        sparsity_x, sparsity_y = compute_cdf(model)
        weights_cdf_x.append(sparsity_x)
        weights_cdf_y.append(sparsity_y)

        is_best = max_loss > current_loss
        max_loss = min(max_loss, current_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, is_best, current_loss, acc1, acc5, train_dir, original_model_acc1, pruned_model_acc1, pruned_threshold, run_id )

        print('Current loss : ', current_loss, ' Max loss : ' ,max_loss)
    
    
    
    model_path= train_dir+'BEST_' + config.filename
    lossarr = np.hstack(lossarr)
    np.save(train_dir+"Densenet_cifar100_loss_"+run_id,lossarr)
    np.save(train_dir+"Densenet_cifar100_L1___"+run_id,l1_loss)
    np.save(train_dir+"Densenet_cifar100_acc1_"+run_id,testarr1)
    np.save(train_dir+"Densenet_cifar100_acc5_"+run_id,testarr5)
    np.save(train_dir+"Densenet_cifar100_cdfx_"+run_id,weights_cdf_x)
    np.save(train_dir+"Densenet_cifar100_cdfy_"+run_id,weights_cdf_y)

    #plot sparsity, training loss, top-1 accuracy and top-5 accuracy
    plot_learning_curve(model,data=model, xlabel="Weights Value", ylabel="CDF", filename=train_dir+"plot_cdf_"+run_id+".png", cdf_data=True)
    plot_learning_curve(model,data=np.log(lossarr), xlabel="Iteration", ylabel="Training Loss", filename=train_dir+"plot_loss_"+run_id+".png")
    plot_learning_curve(model,data=testarr1, xlabel="Epoch", ylabel="Top-1 Accuracy", filename=train_dir+"plot_acc1_"+run_id+".png", ylim=[0, 100])
    plot_learning_curve(model,data=testarr5, xlabel="Epoch", ylabel="Top-5 Accuracy", filename=train_dir+"plot_acc5_"+run_id+".png", ylim=[0, 100])

    return model_path, acc1

def objective(trial):

    #Build model
    if config.network=='densenet':
        model = densenet201()
    elif config.network=='mlp':
        model= MLP()
  
    model = model.cuda()
    
    #Hyperparameters values using Optuna Library
    rho_decay=0.5 #trial.suggest_float('rho_decay', 0.5, 0.6, log=True)
    epsilon_decay=0.56 #trial.suggest_float('epsilon_decay', rho_decay, 0.6, log=True)
    rho= trial.suggest_float('rho', 0.5, 1, log=True) #Suggests float value in range of 0.5 to 1.
    epsilon= trial.suggest_float('epsilon', 0.01, 0.3, log=True) #Suggests float value in range of 0.01 to 0.3.

    run_id = "epsilon_"+str(epsilon)+"_epsilon_decay_"+str(epsilon_decay)+"_rho_"+str(rho)+"_rho_decay_"+str(rho_decay)+"_mu_"+str(config.mu)
    result_dir= config.result_dir+'experiment_'+config.network+'_'+config.dataset+'_'+run_id+'/'

    #Initialization: save the initial weights and do an initial accuracy run
    if not path.exists(path.dirname(result_dir)):   # Check if folder/Path for experiment result exists
        makedirs(path.dirname(result_dir))          # If not, then create one   

    #Set logger
    logger = set_logger(logger_name=result_dir+"logging_"+run_id)
    logger.info("epsilon: "+str(epsilon)+", epsilon_decay: "+str(epsilon_decay)+", rho: "+str(rho)+", rho_decay: "+str(rho_decay)+", mu: "+str(config.mu))
    logger.info("Initial accuracies:")
    
    #Initialize the optimizer with the given parameters
    optimizer = ProxSGD(model.parameters(), epsilon=epsilon, epsilon_decay=epsilon_decay, rho=rho, rho_decay=rho_decay, mu=config.mu)
    
    #Train the model using ProxSGD 
    trained_model_path,trained_accuracy=train_dense201(model,optimizer,run_id, result_dir,logger, 0 , 0 , 0 , retrain=False)
    checkpoint = torch.load(trained_model_path)
    original_model=checkpoint['model']
    print('\nOriginal Model Evaluation: ')
    original_model_loss, original_model_acc1, original_model_acc5, original_model_compression_rate = test(original_model,test_loader,logger)
    
    #Prune trained model with specified pruning threshold
    pruned_threshold= 2.0
    pruned_model=prune_model(original_model,pruned_threshold)
    print('\nPruned Model Evaluation for Threshold value: ', pruned_threshold)
    pruned_model_loss, pruned_model_acc1, pruned_model_acc5 ,pruned_compression_rate= test(pruned_model,test_loader,logger)
    
    result_accuracies_for_each_run=[run_id]
    result_accuracies_for_each_run.append(original_model_acc1)
    result_accuracies_for_each_run.append(pruned_compression_rate)

        
    with open(config.accuracy_comparison_dir+config.csv_filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(result_accuracies_for_each_run)
        f.close()
    

    return original_model_acc1
    

    

if __name__ == "__main__":

    #load Dataset

    if config.dataset=='CIFAR100':
        train_data, test_loader = load_cifar100()  
    elif config.dataset=='MNIST':
        train_data, test_loader = load_mnist()
    

    if not path.exists(path.dirname(config.accuracy_comparison_dir)):   # Check if folder/Path for experiment result exists
        makedirs(path.dirname(config.accuracy_comparison_dir))          # If not, then create one   
    
    with open(config.accuracy_comparison_dir+config.csv_filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['Run_id','Original_Accuracy','compression_rate_2.0']) # Write column names to the csv file.
        f.close()              
    
    storage = optuna.storages.RDBStorage(url="sqlite:///"+config.study_name+".db",engine_kwargs={'pool_pre_ping': True,'connect_args': {'timeout': 10}}) 

    study = optuna.create_study(study_name=config.study_name,storage=storage,direction='maximize', load_if_exists=True,sampler=RandomSampler()) #Create study to maintain history in specified storage and other settings.
    study = optuna.load_study(study_name=config.study_name, storage="sqlite:///"+config.study_name+".db")   #Load existing study to use as history
    
    study.optimize(objective, n_trials=config.n_trials)   #Optimize objective function in the direction we specified while creating study.

    trial = study.best_trial   #save best trial among all trials.

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    print(trial.user_attrs)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_slice(study)
    optuna.visualization.plot_contour(study, params=['rho_decay', 'epsilon_decay','rho'])

    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)