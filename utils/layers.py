import torch
import torch.nn as nn
import numpy as np
from typing import Any,Dict,Optional,Sequence,Tuple,Type,Union,Callable

ModuleType = Type[nn.Module]
def mlp_block(input_dim: int,
              output_dim: int,
              normalize: Optional[Union[nn.BatchNorm1d,nn.LayerNorm]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[torch.Tensor],torch.Tensor]] = None,
              device: Optional[Union[str,int,torch.device]] = None) -> Sequence[ModuleType]:
    block = []
    linear = nn.Linear(input_dim,output_dim,device=device)
    if initialize is not None:
        initialize(linear.weight)
    block.append(linear)
    if normalize is not None:
        block.append(normalize(output_dim,device=device))
    if activation is not None:
        block.append(activation())
    return block, (output_dim,)


def cnn_block(input_shape:Sequence[int],
              filter:int,
              kernel_size:int,
              stride:int,
              normalize: Optional[Union[nn.BatchNorm2d,nn.LayerNorm,nn.GroupNorm,nn.InstanceNorm2d]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[torch.Tensor],torch.Tensor]] = None,
              device: Optional[Union[str,int,torch.device]] = None
              ) -> Sequence[ModuleType]:
    assert len(input_shape) == 3 # CxHxW
    C,H,W = input_shape 
    padding = int((kernel_size-stride)//2)
    block = []
    cnn = nn.Conv2d(C,filter,kernel_size,stride,padding=padding,device=device)
    if initialize is not None:
        initialize(cnn.weight)
    block.append(cnn)
    C = filter
    H = int((H + 2*padding - (kernel_size-1)-1) / stride + 1)
    W = int((W + 2*padding - (kernel_size-1)-1) / stride + 1)
    if normalize is not None:
        if normalize == nn.GroupNorm:
            block.append(normalize(C//2,C,device=device))
        elif normalize == nn.LayerNorm:
            block.append(normalize((C,H,W),device=device))
        else:
            block.append(normalize(C,device=device))
    if activation is not None:
        block.append(activation())
    return block,(C,H,W)

def decnn_block(input_shape: Sequence[int],
                filter: int,
                kernel_size: int,
                stride: int,
                normalize: Optional[Union[nn.BatchNorm2d,nn.LayerNorm,nn.GroupNorm,nn.InstanceNorm2d]] = None,
                activation: Optional[ModuleType] = None,
                initialize: Optional[Callable[[torch.Tensor],torch.Tensor]] = None,
                device: Optional[Union[str,int,torch.device]] = None
                ) -> Sequence[ModuleType]:
    assert len(input_shape) == 3 # CxHxW
    C,H,W = input_shape 
    padding = int((kernel_size-stride+1)//2)
    output_padding = (kernel_size - stride)%2
    if stride == 1:
        output_padding = 0
    block = []
    decnn = nn.ConvTranspose2d(C,filter,kernel_size,stride,padding=padding,output_padding=output_padding,device=device)
    if initialize is not None:
        initialize(decnn.weight)
    block.append(decnn)
    C = filter
    H = (H-1)*stride + kernel_size - 2*padding + output_padding 
    W = (W-1)*stride + kernel_size - 2*padding + output_padding 
    if normalize is not None:
        if normalize == nn.GroupNorm:
            block.append(normalize(C//2,C,device=device))
        elif normalize == nn.LayerNorm:
            block.append(normalize((C,H,W),device=device))
        else:
            block.append(normalize(C,device=device))
    if activation is not None:
        block.append(activation())
    return block,(C,H,W)

def gru_block(input_dim: Sequence[int],
              output_dim: int,
              dropout: float = 0,
              initialize: Optional[Callable[[torch.Tensor],torch.Tensor]] = None,
              device: Optional[Union[str,int,torch.device]] = None) -> ModuleType:
    gru = nn.GRU(input_size=input_dim,
                 hidden_size=output_dim,
                 batch_first=True,
                 dropout=dropout,
                 device=device)
    if initialize is not None:
        for weight_list in gru.all_weights:
            for weight in weight_list:
                if len(weight.shape) > 1:
                    initialize(weight)
    return gru

def lstm_block(input_dim: Sequence[int],
               output_dim: int,
               dropout: float = 0,
               initialize: Optional[Callable[[torch.Tensor],torch.Tensor]] = None,
               device: Optional[Union[str,int,torch.device]] = None) -> ModuleType:
    lstm = nn.LSTM(input_size=input_dim,
                   hidden_size=output_dim,
                   batch_first=True,
                   dropout=dropout,
                   device=device)
    if initialize is not None:
        for weight_list in lstm.all_weights:
            for weight in weight_list:
                if len(weight.shape) > 1:
                    initialize(weight)
    return lstm

def pooling_block(input_shape: Sequence[int],
                  scale:int,
                  pooling:Union[nn.AdaptiveMaxPool2d,nn.AdaptiveAvgPool2d]) -> Sequence[ModuleType]:
    assert len(input_shape) == 3 # CxHxW
    C,H,W = input_shape
    return [pooling(output_size=(H//scale,W//scale))]

def adaptive_pooling_block(output_size:Sequence[int],
                           pooling:Union[nn.AdaptiveAvgPool2d,nn.AdaptiveMaxPool2d]):
    return [pooling(output_size)]
