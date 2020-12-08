dataset='CIFAR100'                                                        # Dataset to used whether CIFAR100 or MNIST
result_dir= 'results_tpes/'                                               # Result directory to save all the experiments
network='densenet'                                                        # Network to be trained densenet/mlp
filename = 'checkpoint.pth.tar'                                           # Filename to save trained model
batch_size=128                                                            # Batch size 
train_epochs=100                                                          # Training Epochs
retrain_epochs=60                                                         # Number of epochs network retrained
threshold= [1.6,2.0,2.5]                                                  # Pruning Threshold. Prune weights when weights are less than corresponding layer's Standarad deviation * threshold 
mu=1e-4                                                                   # Regularization Constant
weight_reg = None                                                         # Remove weight regulatization in Pytorch as ProxSGD already has it implemented
n_trials=60                                                               # Number of trials to search hyperparameter
csv_filename='accuracy_comparison_basedon_thresholds.csv'                 # csv filename to save original model accuracy along with retraining accuracies.
study_name='hyperparameter_cma_search'                                  # Database filename to use history of earlier trials.
accuracy_comparison_dir=result_dir+'accuracy_comparison/'+study_name+'/'  # Directory name to save csv file.

