import torch
import torch.nn as nn
import math

class LoRAConv(nn.Module):
  """
  This is a low-rank adapted linear layer that can be used to replace a standard linear layer.
  
  
  Args:
    module: The linear layer module to adapt.
    rank: The rank of the approximation.
    alpha: The alpha parameter.
  """

  def __init__(
    self,
    module: nn.Module,
    # in_dim: int,
    # out_dim: int,
    rank: int = 4,
    alpha: float = 4.0
  ):
    # ensure the module is a convolutional layer
    #assert isinstance(module, nn.Conv2d), "Module must be a convolutional layer."

    super().__init__() # call the __init__() method of the parent class
    self.rank = rank # rank of the approximation
    self.alpha = alpha # alpha parameter
    self.scaling = self.alpha / self.rank # scaling factor
    self.in_channels = module.in_channels # number of input channels
    self.out_channels = module.out_channels # number of output channels
    self.kernel_size = module.kernel_size # kernel size
    self.stride = module.stride # stride
    self.padding = module.padding # padding

    # make sure that rank is at least 1
    assert self.rank >= 1, "Rank must be at least 1."

    # recreate the convolutional layer and freeze it
    # note: we will copy over the pretrained weights after initializing
    if isinstance(module, ( nn.ConvTranspose2d)):
       self.pretrained = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False)
    else:
       self.pretrained = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False)
    self.pretrained.weight = nn.Parameter(module.weight.detach().clone())
    #self.pretrained.bias = nn.Parameter(module.bias.detach().clone())
    self.pretrained.weight.requires_grad = False # freeze the weights
    #self.pretrained.bias.requires_grad = False # freeze the bias

    # create A and initialize with Kaiming
    self.A = nn.Conv2d(self.in_channels, self.rank, self.kernel_size, self.stride, self.padding, bias=False)
    nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))

    # create B and initialize with zeros
    self.B = nn.Conv2d(self.rank, self.out_channels, 1, 1, 0, bias=False)
    nn.init.zeros_(self.B.weight)

    # ensure that the weights in A and B are trainable
    self.A.weight.requires_grad = True
    self.B.weight.requires_grad = True

  def forward(self, x: torch.Tensor):
        """
        Perform the forward pass of the layer.

        Args:
            x: The input tensor.
        """
        
        pretrained_out = self.pretrained(x) # get the pretrained weights
        lora_out = self.A(x) # apply A
        lora_out = self.B(lora_out) # apply B
        lora_out = lora_out * self.scaling
        return pretrained_out + lora_out

class LoRALinear(nn.Module):

  def __init__(
      self,
      module: nn.Module,
      rank: int = 16,
      alpha: float = 4.0
      ):
      assert isinstance(module, nn.Linear)

      super().__init__()
      self.rank = rank
      self.alpha = alpha
      self.in_dim = module.in_features
      self.out_dim = module.out_features
      self.scaling = self.alpha / self.rank

      assert self.rank >= 1, "Rank"

      self.pretrained = nn.Linear(self.in_dim, self.out_dim, bias=False)
      self.pretrained.weight = nn.Parameter(module.weight.detach().clone())
      self.pretrained.weight.requires_grad = False

      self.A = nn.Linear(self.in_dim, rank, bias=False)
      self.B = nn.Linear(rank, self.out_dim, bias=False)

      nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
      nn.init.zeros_(self.B.weight)

      self.A.weight.requires_grad = True
      self.B.weight.requires_grad = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      '''
      '''
      pretrained_out = self.pretrained(x) # get the pretrained weights
      lora_out = self.A(x) # apply A
      lora_out = self.B(lora_out) # apply B
      lora_out = lora_out * self.scaling
      return pretrained_out + lora_out


def freeze_parameters(model: nn.Module):
  """
  Freeze all parameters in the model.

  Args:
    model: The model to freeze the parameters of.
  """
  for param in model.parameters(): # iterate over the parameters of the model
    param.requires_grad = False # freeze the parameter

def unfreeze_parameters(model: nn.Module):
  """
  Unfreeze all parameters in the model.

  Args:
    model: The model to unfreeze the parameters of.
  """
  for param in model.parameters(): # iterate over the parameters of the model
    param.requires_grad = True # unfreeze the parameter


import copy


def get_updated_model(model: nn.Module, rank: int = 4, alpha: float = 4.0, device: str = 'cuda'):
    """
    Returns a new model with all linear layers replaced by LoRALinear layers.

    Args:
        model: The original model.
        rank: The rank of the approximation.
        alpha: The alpha parameter.
    """
    new_model = copy.deepcopy(model)
    update_model(new_model, rank, alpha, device)
    return new_model


# create a function to replace all linear layers in the the net with LoRALinear layers
def update_model(model: nn.Module, rank: int = 4, alpha: float = 4.0, device: str = 'cuda'):
  """
  Replaces all linear layers in the model with LoRALinear layers.
  
  Args:
    model: The model to update.
    rank: The rank of the approximation.
    alpha: The alpha parameter.
  """
  # make sure there are no LoRALinear layers in the model; return if there are
  for name, module in model.named_modules():
    if isinstance(module, LoRAConv):
      print("Model already contains LoRAConv layers.")
      return
    if isinstance(module, LoRALinear):
      print("Model already contains LoRAConv layers.")
      return
      
  freeze_parameters(model) # freeze all parameters in the model

  for name, module in model.named_children(): # iterate over the children of the model
    
    if "temb"  not in name and "conv_out" not in name: 
      #if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)): # if the module is a linear layer
      #  setattr(model, name, LoRAConv(module, rank, alpha)) # replace it with a LoRALinear layer
      #  print(f"Replaced {name} with LoRALinear layer.")
      if isinstance(module, nn.Linear):
        setattr(model, name, LoRALinear(module, rank, alpha)) # replace it with a LoRALinear layer
        print(f"Replaced {name} with LoRALinear layer.")
      else: # otherwise
        update_model(module, rank, alpha) # recursively call the function on the module

  # move the model to the device
  model.to(device)

  # ensure low-rank matrices are trainable
  for name, module in model.named_modules():
    if isinstance(module, LoRAConv):
      module.A.weight.requires_grad = True
      module.B.weight.requires_grad = True
    if isinstance(module, (nn.BatchNorm2d,nn.BatchNorm1d, nn.LayerNorm)):
      module.weight.requires_grad = True
