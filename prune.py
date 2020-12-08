#This file is used to prune the model based on quality parameter which decides the threshold value by multiplying it with standard deviation of each layer.

import torch
import torch.nn as nn

def prune_model(model,quality_parameter):
    for index,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            weight_copy = module.weight.data.abs().clone()
            std=torch.std(weight_copy)
            threshold= std * quality_parameter
            #threshold= 1e-8
            mask = weight_copy.gt(threshold).float().cuda()
            module.weight.data.mul_(mask)
    return model