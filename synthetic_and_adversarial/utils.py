import random
import numpy as np
import torch
from typing import Iterator
from easydict import EasyDict
import torch
from optimizer.zo import (
    ZoVanilla, 
    ZoAR,
)
from optimizer.zo_hess import (
    ZoHessVanilla,
    ZoARHess,
    HiZOO
)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(
    name: str,
    params: Iterator[torch.Tensor],
    args: EasyDict
) -> torch.optim.Optimizer:
    """
    Get the optimizer class based on the name.
    """
    
    if name == "zo":
        return ZoVanilla(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule)
    elif name == "zoar":
        return ZoAR(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, num_histories=args.num_histories, update_rule=args.update_rule)
    elif name == "zoar_0":
        return ZoAR(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, num_histories=0, update_rule=args.update_rule)
    elif name == "zovh":
        return ZoHessVanilla(params=params, lr=args.lr, num_queries=args.num_queries, mu=args.mu, lambda_hess=args.lambda_hess, orthogonal=False)
    elif name == "zovh_reuse":
        return ZoARHess(params=params, lr=args.lr, num_queries=args.num_queries, mu=args.mu, lambda_hess=args.lambda_hess, num_histories=args.num_histories, orthogonal=False)
    elif name == "hizoo":
        return HiZOO(params=params, lr=args.lr, betas=args.betas, epsilon=args.epsilon, num_queries=args.num_queries, mu=args.mu, update_rule=args.update_rule)
    else:
        raise ValueError(f"Unknown optimizer name: {name}, available optimizers are: fo, zo, zoar, zoar_0")