import torch
import torch.nn as nn
import pdb


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t, **kargs):
        output = self.model(x, cond, t, **kargs)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args, **kargs):
        x.requires_grad_()
        y = self(x, *args, **kargs)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad

