import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F


class AdaptiveFusion(nn.Module):
    """ For a specific category, adaptively fuse the instance prototype to get the class prototype """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.temperature = np.power(self.d_model, 0.5)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, query, each_class_instance_prototypes):
        # query = (batchsize, len_q, d_model)
        batchsize, _, _ = query.shape
        
        # each_class_instance_prototypes = (num_instance, d_model) -> (batchsize, num_instance, d_model)
        each_class_instance_prototypes = each_class_instance_prototypes.expand(batchsize, -1, -1)
        logits = torch.bmm(query, each_class_instance_prototypes.transpose(1,2))
        logits = logits / self.temperature # logits = (batchsize, len_q, num_instance)
        weights = logits.sum(dim=1, keepdim=True) # importance = (batchsize, 1, num_instance)
        weights = self.softmax(weights)
        class_prototype = torch.bmm(weights, each_class_instance_prototypes) # class_prototype = (batchsize, 1, d_model)
        return class_prototype

        
