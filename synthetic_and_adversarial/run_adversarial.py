import argparse
import os
import time
import tqdm
import time
import torch
from model.attack import Attack
from utils import set_seed, get_optimizer
from easydict import EasyDict
import yaml

def train(model, optimizer, args):
    history = []

    history.append(model().item())
    
    for _ in tqdm.tqdm(range(args.num_iterations), desc="Training", leave=False, disable=False):
        optimizer.zero_grad()
        f = optimizer.step(model)
        
        if isinstance(f, torch.Tensor):
            f = f.item()
        history.append(f)
    
    return history

def main(args):
    seed = args.seed
    model_name = args.model
    dataset_name = args.dataset
    dimension = args.x_dim
    idx = args.idx
    optimizers = args.optimizers
    device = args.device
    
    device = torch.device(device)
    print(f"Using device: {device}")

    histories = []
    start = time.time()
    for optimizer_name in optimizers:
        for run in range(args.num_runs):
            set_seed(seed + run)

            x_init = torch.randn(dimension, device=device)
            model = Attack(x_init, idx=idx)
            
            valid_parameters = [p for n, p in model.named_parameters() if n == "x"]

            optimizer = get_optimizer(optimizer_name, valid_parameters, args)

            start_1 = time.time()
            history = train(model, optimizer, args)

            print(f"{optimizer_name} (seed {seed + run}) optimized value: {history[-1]}, Time taken: {time.time() - start_1:.2f} seconds")

            histories.append(history)

            # Save the history to a file
            tag = f"{args.dataset}_{optimizer_name}_{args.update_rule}_ni{args.num_iterations}_lr{args.lr}_nq{args.num_queries}_mu{args.mu}_nh{args.num_histories}_s{seed + run}"
            os.makedirs("results/attack", exist_ok=True)
            torch.save(history, f"results/attack/{tag}.pt")

    print(f"Total Time taken: {time.time() - start:.2f} seconds")

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Hessian Approximation on Black-Box Adversarial Attacks Task')
    parser.add_argument('--config', type=str, default='config/adversarial.yaml', help='Path to the config file')
    path_to_config = parser.parse_args().config
    with open(path_to_config, 'r') as f:
        args = yaml.safe_load(f)
    args = EasyDict(args)

    main(args)
