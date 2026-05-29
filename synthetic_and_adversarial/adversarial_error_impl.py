"""
Utilities to compute true and approximate Hessians for the adversarial attack task,
log per-step Frobenius errors to CSV, and run quick tests from a YAML config.

Used by scripts and analyses under `synthetic_and_adversarial/`.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from easydict import EasyDict
import os
import csv
from pathlib import Path
from model.attack import Attack
from utils import get_optimizer, set_seed
from hessian_approx_ import (
    hessian_CD_approx,
    hessian_S1_approx,
    hessian_S2_approx,
    hessian_S3_approx,
    hessian_ZoVH_approx,
    hessian_ZoVH_reuse_approx
)


class AttackFunctionWrapper:
    """
    Provide f(theta, xi) API for the Attack objective required by hessian_approx methods.

    - theta: numpy array of shape (d,)
    - xi: numpy array (unused; kept for API compatibility)
    Returns the scalar value of the objective at theta.
    """

    def __init__(self, attack: Attack):
        self.attack = attack
        self.d = attack.dim
        self.size = attack.size
        self.device = attack.device
        self.dtype = attack.x.dtype

    def f(self, theta: np.ndarray, xi: np.ndarray) -> float:
        x = torch.tensor(theta, dtype=self.dtype, device=self.device).reshape(*self.size)
        new_image = torch.clamp(x, self.attack.lb, self.attack.ub) + self.attack.data
        new_image = torch.clamp(new_image, 0, 1)
        target_label = int(((self.attack.target + 1) % 10))
        loss = self.attack.get_loss(new_image, target_label, targeted=True)
        return float((-loss).item())


def compute_true_hessian_for_attack(attack: Attack) -> np.ndarray:
    """
    Compute the exact Hessian of the attack objective w.r.t. attack.x via autograd.
    Returns a (d, d) numpy array.
    """

    def functional_loss(x_vec: torch.Tensor) -> torch.Tensor:
        x = x_vec.reshape(*attack.size)
        new_image = torch.clamp(x, attack.lb, attack.ub) + attack.data
        new_image = torch.clamp(new_image, 0, 1)
        # Straight-through like Attack.forward so grads flow through x
        new_image = x + (new_image - x).detach()
        target_label = int(((attack.target + 1) % 10))
        loss = attack.get_loss(new_image, target_label, targeted=True)
        return -loss

    x0 = attack.x.detach().clone().requires_grad_(True)
    try:
        H = torch.autograd.functional.hessian(functional_loss, x0, vectorize=True)
    except TypeError:
        # Fallback for older PyTorch versions
        H = torch.autograd.functional.hessian(functional_loss, x0)
    return H.detach().cpu().numpy()


def compute_hessian_approximations(
    theta_np: np.ndarray,
    wrapper: AttackFunctionWrapper,
    d: int,
    K: int,
    mu: float,
    noise_std: float,
    sample_seed: int,
    history_values: Dict,
) -> Dict[str, np.ndarray]:
    """
    Run all Hessian approximation methods for the current theta.
    """
    approx: Dict[str, np.ndarray] = {}
    approx["CD"] = hessian_CD_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["S1"] = hessian_S1_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["S2"] = hessian_S2_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["S3"] = hessian_S3_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["ZoVH"] = hessian_ZoVH_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["ZoVH_reuse"] = hessian_ZoVH_reuse_approx(
        theta_np, wrapper, d, K, mu, noise_std, history_values, sample_seed
    )
    return approx


def frobenius_error(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord="fro"))


# Mapping from internal method keys to (output filename, CSV Method value)
METHOD_FILE_MAP = {
    "CD": ("CD.csv", "CD"),
    "S1": ("S1.csv", "S1"),
    "S2": ("S2.csv", "S2"),
    "S3": ("S3.csv", "S3"),
    "ZoVH": ("ZoVH.csv", "ZoVH"),
    "ZoVH_reuse": ("ZoVH_reuse.csv", "ZoVH_reuse"),
}


def _get_output_dir() -> Path:
    """Return the adversarial results directory path."""
    return Path(__file__).resolve().parent / "NEW result adversarial"


def _ensure_csv_header(fp: Path) -> None:
    """Create the CSV with header if it doesn't exist or is empty."""
    if not fp.exists() or fp.stat().st_size == 0:
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Function",
                    "Dimension",
                    "Seed",
                    "Method",
                    "Sample_Index",
                    "Sample_RandomSeed",
                    "Frobenius_Error",
                ]
            )


def _append_result_row(
    function_name: str,
    dimension: int,
    seed: int,
    method_key: str,
    sample_index: int,
    sample_random_seed: int,
    fro_error: float,
) -> None:
    """Append a single result row to the per-method CSV."""
    if method_key not in METHOD_FILE_MAP:
        return
    filename, method_csv_name = METHOD_FILE_MAP[method_key]
    out_dir = _get_output_dir()
    fp = out_dir / filename
    _ensure_csv_header(fp)
    with fp.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                function_name,
                int(dimension),
                int(seed),
                method_csv_name,
                int(sample_index),
                int(sample_random_seed),
                float(fro_error),
            ]
        )


def run_hessian_error_experiment(args: EasyDict) -> Dict[Tuple[str, int], Dict[str, Dict[str, object]]]:
    """
    For each optimizer and run, iterate the adversarial optimization and at each step:
      - compute true Hessian
      - estimate approximate Hessians by all methods
      - record Frobenius errors

    Maintains history for the reuse-based estimator via (theta, seed).

    Returns:
      dict keyed by (optimizer_name, run_idx) -> { method_name: { 'mean', 'std', 'per_step' } }
    """
    device = torch.device(args.device)

    results_all: Dict[Tuple[str, int], Dict[str, Dict[str, object]]] = {}

    for optimizer_name in args.optimizers:
        for run in range(args.num_runs):
            set_seed(args.seed + run)
            run_seed = int(args.seed + run)

            x_init = torch.randn(args.x_dim, device=device)
            attack = Attack(x_init, idx=args.idx)
            attack.to(device)
            attack.eval()

            valid_parameters = [p for n, p in attack.named_parameters() if n == "x"]
            optimizer = get_optimizer(optimizer_name, valid_parameters, args)

            wrapper = AttackFunctionWrapper(attack)

            historical_thetas: List[np.ndarray] = []
            historical_seeds: List[int] = []

            per_step_errors: Dict[str, List[float]] = {
                "CD": [],
                "S1": [],
                "S2": [],
                "S3": [],
                "ZoVH": [],
                "ZoVH_reuse": [],
            }

            start_time = time.time()
            for sample_index in range(args.num_iterations):
                # 1) True Hessian at current x
                H_true = compute_true_hessian_for_attack(attack)

                # 2) Approximations
                theta_np = attack.x.detach().cpu().numpy()
                sample_seed = int(np.random.randint(0, 2**31 - 1))
                history_values = {
                    "thetas": historical_thetas.copy(),
                    "historical_randomseeds": historical_seeds.copy(),
                }
                approx_hess = compute_hessian_approximations(
                    theta_np=theta_np,
                    wrapper=wrapper,
                    d=args.x_dim,
                    K=args.num_queries,
                    mu=args.mu,
                    noise_std=0.0,
                    sample_seed=sample_seed,
                    history_values=history_values,
                )

                # 3) Errors (+ write to CSV)
                for name, H_hat in approx_hess.items():
                    err = frobenius_error(H_true, H_hat)
                    per_step_errors[name].append(err)
                    _append_result_row(
                        function_name="adversarial",
                        dimension=int(args.x_dim),
                        seed=run_seed,
                        method_key=name,
                        sample_index=sample_index,
                        sample_random_seed=sample_seed,
                        fro_error=float(err),
                    )

                # Update history for reuse (store current theta & seed as most recent)
                historical_thetas.insert(0, theta_np.copy())
                historical_seeds.insert(0, sample_seed)
                max_hist = int(getattr(args, "num_histories", 0))
                if max_hist > 0:
                    historical_thetas = historical_thetas[:max_hist]
                    historical_seeds = historical_seeds[:max_hist]

                # Advance optimization one step
                optimizer.zero_grad()
                _ = optimizer.step(attack)

            elapsed = time.time() - start_time

            # Aggregate statistics
            stats = {
                key: {
                    "mean": float(np.mean(vals)) if len(vals) > 0 else float("nan"),
                    "std": float(np.std(vals)) if len(vals) > 0 else float("nan"),
                    "per_step": vals,
                }
                for key, vals in per_step_errors.items()
            }

            print(f"[Hessian Error] {optimizer_name} (seed {args.seed + run}) finished in {elapsed:.2f}s")
            for k, v in stats.items():
                print(f"  - {k}: mean={v['mean']:.6e}, std={v['std']:.6e}")

            results_all[(optimizer_name, run)] = stats

    return results_all


def test_adversarial_error_small(config_path: str = "config/adversarial.yaml"):
    """
    Quick test that reduces iterations/queries for speed.
    """
    import yaml

    with open(config_path, "r") as f:
        args = yaml.safe_load(f)
    args = EasyDict(args)

    # shrink for speed
    args.num_iterations = min(2, int(args.num_iterations))
    args.num_queries = min(5, int(args.num_queries))
    args.optimizers = args.optimizers[:1] if len(args.optimizers) > 0 else ["zoar"]

    results = run_hessian_error_experiment(args)
    return results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Compute Hessian approximation errors on adversarial attack task")
    parser.add_argument("--config", type=str, default="config/adversarial.yaml", help="Path to the config file")
    parser.add_argument("--quick", action="store_true", help="Run a quick test (few iters)")
    cli_args = parser.parse_args()

    with open(cli_args.config, "r") as f:
        args = yaml.safe_load(f)
    args = EasyDict(args)

    if cli_args.quick:
        _ = test_adversarial_error_small(cli_args.config)
    else:
        _ = run_hessian_error_experiment(args)
