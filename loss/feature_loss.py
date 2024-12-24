from turtle import forward
import torch.nn as nn
from torch.nn import functional as F

import torch

class feature_loss(nn.Module):
    def __init__(self):
        super(feature_loss, self).__init__()
    
    def forward(self, student_feature, teacher_feature):
        return (self.at(student_feature) - self.at(teacher_feature)).pow(2).mean()

    def at(self, f):
        return  f
        return F.normalize(f.pow(2).mean(1).view(f.size(0), -1))