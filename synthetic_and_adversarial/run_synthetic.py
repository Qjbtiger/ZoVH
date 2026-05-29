"""
Run synthetic function optimization with various (zero/first)-order optimizers.

- Reads configuration from YAML (default: config/synthetic.yaml)
- Saves objective histories to results/synthetic/*.pt
- See README for details and plotting instructions.
"""

import argparse
import os
import time
import tqdm
import torch
from model.synthetic_functions import get_synthetic_funcs
from utils import set_seed, get_optimizer
from easydict import EasyDict

import yaml

def train(func, optimizer, args):
    history = []
    with torch.no_grad():
        history.append(func().item())
    
    for _ in tqdm.tqdm(range(args.num_iterations), desc="Training", leave=False, disable=False):
        optimizer.zero_grad()
        f = optimizer.step(func)
        
        if isinstance(f, torch.Tensor):
            f = f.item()
        history.append(f)
    
    return history

def main(args):
    seed = args.seed
    func_name_list = args.func_name_list
    dimension = args.dimension
    optimizers = args.optimizers
    
    device = torch.device("cpu")
    print(f"Using device: {device}")

    histories = []
    start = time.time()
    for func_name in func_name_list:
        for optimizer_name in optimizers:
            for run in range(args.num_runs):
                set_seed(seed + run)

                x_init = torch.randn(dimension)

                func = get_synthetic_funcs(func_name, x_init).to(device)
                func.eval()
                optimizer = get_optimizer(optimizer_name, func.parameters(), args)

                start_1 = time.time()

                history = train(func, optimizer, args)

                print(f"{func_name} + {optimizer_name} (seed {seed + run}) optimized value: {history[-1]}, Time taken: {time.time() - start_1:.2f} seconds")

                histories.append(history)

                # Save the history to a file
                tag = f"{func_name}_{optimizer_name}_{args.update_rule}_d{dimension}_ni{args.num_iterations}_lr{args.lr}_nq{args.num_queries}_mu{args.mu}_nh{args.num_histories}_s{seed + run}"
                os.makedirs("results/synthetic", exist_ok=True)
                torch.save(history, f"results/synthetic/{tag}.pt")

    print(f"Total Time taken: {time.time() - start:.2f} seconds")

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Hessian Approximation on Synthetic Functions Optimization')
    parser.add_argument('--config', type=str, default='config/synthetic.yaml', help='Path to the config file')
    path_to_config = parser.parse_args().config
    with open(path_to_config, 'r') as f:
        args = yaml.safe_load(f)
    args = EasyDict(args)

    main(args)
