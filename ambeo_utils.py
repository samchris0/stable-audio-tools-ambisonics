import torch

from torch import nn

class Ambeo4(nn.Module):
    def __call__(self, signal):
        signal_shape = signal.shape
            # only allow inputs with 4 or more channels
        if len(signal_shape) >= 4: 
            signal = signal[:4, :]  
        else:
            raise ValueError("Ambeo4 can only be called on a signal with four or more channels")