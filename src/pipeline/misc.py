import gc
import numpy as np
from math import ceil

import torch
import torch.nn as nn

from numpy import prod as prod
from numpy.random import choice as choice
from random import seed

from typing import List, Tuple, Optional, Callable

def exists(var):
    return var is not None

def default(var, val):
    return val if var is None else var

def allTrue(idx, name):
    return True

class Info:
    '''
        Very simplistic implementation of PyTorch hooks that are used to grab information about the
        module layer shape and name.
    '''
    def __init__(self) -> None:
        self.shapes = []
        self.names  = []
        self.layers = []

        self.depth = -1
        self.traces = {'conv2d' : 0, 'relu' : 0, 'mpool2d' : 0, 'fc' : 0, 'avgpool2d' : 0}

    def __call__(self, module, inp, out):
        self.shapes += [out.detach().cpu().numpy().squeeze().shape]
        self.names  += [self.get_name(module)]
        self.layers += [module]

    def get_name(self, module : nn.Module) -> str:
        self.depth += 1

        if   isinstance(module, nn.Conv2d):    return f"{self.depth}_conv2d_{self._update('conv2d')}" 
        elif isinstance(module, nn.MaxPool2d): return f"{self.depth}_mpool2d_{self._update('mpool2d')}" 
        elif isinstance(module, nn.ReLU):      return f"{self.depth}_relu_{self._update('relu')}" 
        elif isinstance(module, nn.Linear):    return f"{self.depth}_fc_{self._update('fc')}"
        elif isinstance(module, nn.AdaptiveAvgPool2d): return f"avgpool2d_{self._update('avgpool2d')}"
        else: raise ValueError(f'Unknow module type {module}')

    def _update(self, key : str): 
        self.traces[key] = self.traces[key] + 1
        return str(self.traces[key]).zfill(2)
    
    def __str__(self) -> str:
        # Print the content of the Info object into a string
        return '\n'.join([f'{name} : {shape}' for name, shape in zip(self.names, self.shapes)])

def get_info(
    model : nn.Module,
    inp_shape : Tuple[int],
    exclude : Optional[List[nn.Module]] = None,
    include : Optional[Callable] = None,
    ) -> Info:
    exclude = default(exclude, [type(None)])
    include = default(include, lambda idx, name : True)

    # Get the flattened model, excluding unwanted layer types
    layers = flatten(model)
    names  = generate_hook_keys(layers)

    # Filter layer based on depth and names
    layers = [l for idx, (name, l) in enumerate(zip(names, layers))
                if include(idx, name) and not isinstance(l, tuple(exclude))]

    # Attach a basic recording hook to each target layer
    info = Info()
    hook_handles = [l.register_forward_hook(info) for l in layers]

    # Pass a mock up input into the network to trigger the hooks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    inp = torch.zeros(inp_shape).to(device)

    _ = model(inp)

    # Clean the hooks
    for hook in hook_handles: hook.remove()

    return info

def get_units(
    num_unit : int | None,
    info : Info,
    include : Optional[Callable] = None,
    exclude : Optional[List[nn.Module]] = None,
    ) -> List[np.ndarray]:
    if num_unit is None: return None

    exclude = default(exclude, [type(None)])
    include = default(include, lambda idx, name : True)

    units  = {name : np.unravel_index(choice(prod(shape), replace = False, size = min(num_unit, prod(shape))), shape)
                for idx, (name, shape, layer) in enumerate(zip(info.names, info.shapes, info.layers))
                    if include(idx, name) and not isinstance(layer, tuple(exclude))
            }
        
    return units

def worker_init_fn(worker_id, deterministic : bool = False):
    # Get the dataset
    info = torch.utils.data.get_worker_info()

    dataset = info.dataset
    start = dataset.start
    end   = dataset.end

    if deterministic:
        rid = dataset.dataset.unique_id
        rng = dataset.dataset.generator
        if rng: rng.manual_seed(rid + worker_id)
    
    # Configure the dataset to only process the split workload
    per_worker = int(ceil((end - start) / float(info.num_workers)))
    worker_id = info.id

    dataset.start = start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, end)

def jackknife(*args, axis : int = 0):
    size = np.shape (args[0])[axis]
    idxs = np.arange (size)
    for shift in range (size):
        s_idx = np.roll(idxs, -shift)[:-1]
        yield [arg.take (s_idx, axis = axis) for arg in args]

def generate_hook_keys(layers : List[nn.Module], do_raise : bool = False) -> List[str]:
    hook_keys = []
    names = {'conv2d' : 0, 'relu' : 0, 'mpool2d' : 0, 'fc' : 0, 'avgpool2d' : 0}
    
    def update(key : str): 
        names[key] = names[key] + 1
        return str(names[key]).zfill(2)

    for l, L in enumerate(layers):
        if   isinstance(L, nn.Conv2d):    hook_keys += [f"{l}_conv2d_{update('conv2d')}"] 
        elif isinstance(L, nn.MaxPool2d): hook_keys += [f"{l}_mpool2d_{update('mpool2d')}"] 
        elif isinstance(L, nn.ReLU):      hook_keys += [f"{l}_relu_{update('relu')}"] 
        elif isinstance(L, nn.Linear):    hook_keys += [f"{l}_fc_{update('fc')}"]
        elif isinstance(L, nn.AdaptiveAvgPool2d): hook_keys += [f"{l}_avgpool2d_{update('avgpool2d')}"]
        else: 
            if do_raise: raise ValueError(f'Unknow layer type {L}')
            else: hook_keys += [f'{l}_unknown']

    return hook_keys

def filter_module(
    module : nn.Module,
    exclude : Optional[List[nn.Module]] = None,
    include : Optional[Callable] = None,
) -> List[nn.Module]:
    exclude = default(exclude, [type(None)])
    include = default(include, lambda i, name: True)

    # Get layer names for consistent naming with Info
    flat = flatten(module)
    names = generate_hook_keys(flat)

    layers = [l for i, (l, name) in enumerate(zip(flat, names)) if include(i, name) and not isinstance(l, tuple(exclude))]

    return layers

def flatten(model : nn.Module, exclude : List[nn.Module] = []) -> List[nn.Module]:
    flattened = [flatten(children) for children in model.children()]
    res = [model] if list(model.children()) == [] else []

    for c in flattened: res += c
    
    return res

def from_csv(val) -> float:
    try:
        return float(val)
    
    except ValueError:
        return np.nan

def dump_garbage():
    """
    show us what's the garbage about
    """
        
    # force collection
    print ('\nGARBAGE:')
    gc.collect()

    print ('\nGARBAGE OBJECTS:')
    for x in gc.garbage:
        s = str(x)
        if len(s) > 80: s = s[:80]
        print (type(x), "\n  ", s)

