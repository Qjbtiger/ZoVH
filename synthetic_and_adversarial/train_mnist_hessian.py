import os
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from easydict import EasyDict

from model.cnn import CNN
from hessian_approx_ import (
    hessian_CD_approx,
    hessian_S1_approx,
    hessian_S2_approx,
    hessian_S3_approx,
    hessian_ZoVH_approx,
    hessian_ZoVH_reuse_approx,
)
from utils import set_seed


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
    return Path(__file__).resolve().parent / "NEW result cnn"


def _ensure_csv_header(fp: Path) -> None:
    if not fp.exists() or fp.stat().st_size == 0:
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Function",  # here: CNN layer name, e.g., conv1.weight
                    "Dimension",  # effective dimension (possibly subsampled)
                    "Seed",
                    "Method",
                    "Sample_Index",  # training step index when measured
                    "Sample_RandomSeed",  # random seed used by the estimator per sample
                    "Frobenius_Error",
                ]
            )


def _append_result_row(
    layer_name: str,
    dimension: int,
    seed: int,
    method_key: str,
    sample_index: int,
    sample_random_seed: int,
    fro_error: float,
) -> None:
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
                layer_name,
                int(dimension),
                int(seed),
                method_csv_name,
                int(sample_index),
                int(sample_random_seed),
                float(fro_error),
            ]
        )


def _get_module_and_param(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str, nn.Parameter]:
    """Return (module, param_attr_name, param) given a qualified name like 'conv1.weight'."""
    parts = qualified_name.split(".")
    if len(parts) != 2:
        raise ValueError(f"Only supports names like 'layer.weight'; got {qualified_name}")
    module_name, attr_name = parts
    module = getattr(model, module_name)
    param = getattr(module, attr_name)
    if not isinstance(param, torch.nn.Parameter):
        raise ValueError(f"Attribute {qualified_name} is not an nn.Parameter")
    return module, attr_name, param


class LayerFunctionWrapper:
    """
    f(theta, xi) for a specific model layer and a fixed mini-batch.

    theta is the flattened vector of either the full weight, or a subsampled subset
    specified by 'indices'. The function reconstructs the full weight by inserting
    theta into the proper indices (the rest remain at their current values), runs a
    forward pass on (inputs, targets) and returns the batch loss as a float.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_name: str,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        device: torch.device,
        indices: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.layer_name = layer_name
        self.inputs = inputs
        self.targets = targets
        self.criterion = criterion
        self.device = device
        self.module, self.attr_name, self.param = _get_module_and_param(model, layer_name)

        self.full_shape = self.param.shape
        self.full_numel = self.param.numel()
        if indices is None:
            self.indices = None
            self.d = self.full_numel
        else:
            idx = np.asarray(indices).astype(np.int64)
            if idx.ndim != 1:
                raise ValueError("indices must be a 1-D array")
            if np.max(idx) >= self.full_numel or np.min(idx) < 0:
                raise ValueError("indices out of range for parameter size")
            self.indices = idx
            self.d = int(idx.size)

    @torch.no_grad()
    def _apply_theta_to_model(self, theta_flat_tensor: torch.Tensor) -> torch.nn.Parameter:
        """Create a temporary weight Parameter with theta inserted and set it on the module. Returns the previous Parameter to restore later."""
        prev_param: nn.Parameter = getattr(self.module, self.attr_name)
        # Start from current full weights
        full_flat = prev_param.detach().flatten().clone()
        if self.indices is None:
            full_flat = theta_flat_tensor.detach().clone()
        else:
            full_flat[self.indices] = theta_flat_tensor.detach().clone()
        new_weight = nn.Parameter(full_flat.reshape(self.full_shape))
        setattr(self.module, self.attr_name, new_weight)
        return prev_param

    def f(self, theta: np.ndarray, xi: np.ndarray) -> float:
        theta_t = torch.tensor(theta, dtype=self.param.dtype, device=self.device)
        with torch.no_grad():
            prev = self._apply_theta_to_model(theta_t)
            self.model.eval()  # disable dropout for consistent measurement
            out = self.model(self.inputs)
            loss = self.criterion(out, self.targets)
            # Restore
            setattr(self.module, self.attr_name, prev)
            return float(loss.item())


def compute_true_hessian_for_layer(
    model: nn.Module,
    layer_name: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the exact Hessian of the batch loss w.r.t. the selected layer's weights
    (optionally subsampled by 'indices'). Returns a (d, d) numpy array.
    """

    module, attr_name, param = _get_module_and_param(model, layer_name)
    full_numel = param.numel()
    full_shape = param.shape

    # Build initial theta vector (subsampled or full)
    with torch.no_grad():
        full_flat0 = param.detach().flatten().clone().to(device)
        if indices is None:
            theta0 = full_flat0
        else:
            theta0 = full_flat0[indices]
    theta0 = theta0.detach().clone().requires_grad_(True)

    def loss_wrt_theta(theta_vec: torch.Tensor) -> torch.Tensor:
        # Assemble full weight tensor using theta_vec
        if indices is None:
            full_flat = theta_vec
        else:
            full_flat = full_flat0.clone()
            full_flat[indices] = theta_vec
        weight_param = nn.Parameter(full_flat.reshape(full_shape))
        # Swap into model
        prev_param = getattr(module, attr_name)
        setattr(module, attr_name, weight_param)
        try:
            model.eval()
            out = model(inputs)
            loss = criterion(out, targets)
        finally:
            # Always restore the original Parameter to keep training state intact
            setattr(module, attr_name, prev_param)
        # We return the loss (no negation). Hessian is of training loss.
        return loss

    try:
        H = torch.autograd.functional.hessian(loss_wrt_theta, theta0, vectorize=True)
    except TypeError:
        H = torch.autograd.functional.hessian(loss_wrt_theta, theta0)
    return H.detach().cpu().numpy()


def compute_hessian_approximations(
    wrapper: LayerFunctionWrapper,
    theta_np: np.ndarray,
    d: int,
    K: int,
    mu: float,
    noise_std: float,
    sample_seed: int,
    history: Dict,
) -> Dict[str, np.ndarray]:
    approx: Dict[str, np.ndarray] = {}
    approx["CD"] = hessian_CD_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["S1"] = hessian_S1_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["S2"] = hessian_S2_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["S3"] = hessian_S3_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["ZoVH"] = hessian_ZoVH_approx(theta_np, wrapper, d, K, mu, noise_std, sample_seed)
    approx["ZoVH_reuse"] = hessian_ZoVH_reuse_approx(theta_np, wrapper, d, K, mu, noise_std, history, sample_seed)
    return approx


def frobenius_error(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord="fro"))


def build_dataloaders(args: EasyDict) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(root=args.data_root, train=True, transform=tfm, download=True)
    test_ds = datasets.MNIST(root=args.data_root, train=False, transform=tfm, download=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def maybe_select_indices(total_dim: int, target_dim: int, rng: np.random.RandomState) -> Optional[np.ndarray]:
    if target_dim <= 0 or target_dim >= total_dim:
        return None
    return np.sort(rng.choice(total_dim, size=target_dim, replace=False))


def train_and_log_hessian_errors(config_path: str = "config/cnn_mnist.yaml") -> None:
    import yaml

    with open(config_path, "r") as f:
        args = EasyDict(yaml.safe_load(f))

    device = torch.device(args.device)
    set_seed(args.seed)

    train_loader, test_loader = build_dataloaders(args)

    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.NLLLoss()

    global_step = 0
    rng = np.random.RandomState(args.seed)

    # Prepare per-layer subsampling indices (fixed per run) using dotted names like 'conv1.weight'
    layer_configs = args.layers  # dict of layer_name -> { eval_dim: int }
    layer_indices: Dict[str, Optional[np.ndarray]] = {}
    for layer_name, layer_conf in layer_configs.items():
        module, attr, param = _get_module_and_param(model, layer_name)
        total_dim = int(param.numel())
        target_dim = int(layer_conf.get("eval_dim", total_dim))
        layer_indices[layer_name] = maybe_select_indices(total_dim, target_dim, rng)

    # Maintain history buffers for reuse estimator per layer
    history_buffers: Dict[str, Dict[str, List]] = {
        lname: {"thetas": [], "seeds": []} for lname in layer_configs.keys()
    }
    # Track previous step's theta and seed per layer to ensure reuse uses past info
    last_theta: Dict[str, Optional[np.ndarray]] = {lname: None for lname in layer_configs.keys()}
    last_seed: Dict[str, Optional[int]] = {lname: None for lname in layer_configs.keys()}

    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # Hessian logging
            if (global_step % args.hessian.log_every_steps) == 0:
                for layer_name, layer_conf in layer_configs.items():
                    module, attr, param = _get_module_and_param(model, layer_name)
                    idxs = layer_indices[layer_name]

                    # 0) Move previous step's (theta, seed) into history so reuse has past info now
                    hb = history_buffers[layer_name]
                    if last_theta[layer_name] is not None and last_seed[layer_name] is not None:
                        hb["thetas"].insert(0, last_theta[layer_name])
                        hb["seeds"].insert(0, int(last_seed[layer_name]))
                        max_hist = int(getattr(args.hessian, "num_histories", 0))
                        if max_hist > 0:
                            hb["thetas"] = hb["thetas"][:max_hist]
                            hb["seeds"] = hb["seeds"][:max_hist]

                    # Build wrapper and compute true Hessian (on current batch)
                    wrapper = LayerFunctionWrapper(
                        model=model,
                        layer_name=layer_name,
                        inputs=x,
                        targets=y,
                        criterion=criterion,
                        device=device,
                        indices=idxs,
                    )
                    # theta current
                    with torch.no_grad():
                        full_flat = param.detach().flatten().clone().cpu().numpy()
                        theta_np = full_flat if idxs is None else full_flat[idxs]

                    try:
                        H_true = compute_true_hessian_for_layer(
                            model=model,
                            layer_name=layer_name,
                            inputs=x,
                            targets=y,
                            criterion=criterion,
                            device=device,
                            indices=idxs,
                        )
                    except RuntimeError as e:
                        print(f"[WARN] True Hessian failed for {layer_name} at step {global_step}: {e}")
                        H_true = None

                    # 1) Prepare history snapshot for reuse estimator
                    history = {"thetas": hb["thetas"].copy(), "historical_randomseeds": hb["seeds"].copy()}

                    # Compute approximations only if true Hessian succeeded
                    if H_true is not None:
                        sample_seed = int(rng.randint(0, 2**31 - 1))
                        approx = compute_hessian_approximations(
                            wrapper=wrapper,
                            theta_np=theta_np,
                            d=wrapper.d,
                            K=args.hessian.num_queries,
                            mu=args.hessian.mu,
                            noise_std=args.hessian.noise_std,
                            sample_seed=sample_seed,
                            history=history,
                        )
                        for mname, H_hat in approx.items():
                            err = frobenius_error(H_true, H_hat)
                            _append_result_row(
                                layer_name=layer_name,
                                dimension=int(wrapper.d),
                                seed=int(args.seed),
                                method_key=mname,
                                sample_index=int(global_step),
                                sample_random_seed=int(sample_seed),
                                fro_error=float(err),
                            )

                        # 2) After computing current approximations, cache current (theta, seed)
                        last_theta[layer_name] = theta_np.copy()
                        last_seed[layer_name] = int(sample_seed)

            if (batch_idx + 1) % args.log_every_batches == 0:
                acc = evaluate(model, test_loader, device)
                print(
                    f"Epoch {epoch+1}/{args.num_epochs} Step {batch_idx+1}/{len(train_loader)} "
                    f"Loss {loss.item():.4f} TestAcc {acc:.3f}"
                )

            global_step += 1

    print("Training + Hessian error logging complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN on MNIST and log Hessian approx errors for selected layers")
    parser.add_argument("--config", type=str, default="config/cnn_mnist.yaml")
    args = parser.parse_args()
    train_and_log_hessian_errors(args.config)
