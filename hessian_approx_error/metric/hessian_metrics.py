"""
Hessian Approximation Evaluation Metrics

This module provides various metrics to evaluate the quality of Hessian approximations
by comparing them with the exact Hessian matrix.
"""

import torch
import numpy as np
from typing import Dict, Optional, Union


class HessianMetrics:
    """
    A class to compute various metrics for evaluating Hessian approximations.
    """
    
    def __init__(self, exact_hessian: Union[torch.Tensor, np.ndarray], 
                 approx_hessian: Union[torch.Tensor, np.ndarray]):
        """
        Initialize with exact and approximate Hessian matrices.
        
        Args:
            exact_hessian: The exact/true Hessian matrix
            approx_hessian: The approximated Hessian matrix
        """
        # Convert to torch tensors if numpy arrays
        if isinstance(exact_hessian, np.ndarray):
            exact_hessian = torch.from_numpy(exact_hessian)
        if isinstance(approx_hessian, np.ndarray):
            approx_hessian = torch.from_numpy(approx_hessian)
            
        self.exact_H = exact_hessian.float()
        self.approx_H = approx_hessian.float()
        
        # Verify dimensions match
        assert self.exact_H.shape == self.approx_H.shape, \
            f"Shape mismatch: exact {self.exact_H.shape} vs approx {self.approx_H.shape}"
        
        # Compute error matrix once
        self.error = self.approx_H - self.exact_H
        
    def frobenius_norm_error(self) -> float:
        """
        Compute the Frobenius norm of the error matrix.
        
        ||H_approx - H_exact||_F
        
        Returns:
            Frobenius norm of the error
        """
        return torch.norm(self.error, p='fro').item()
    
    def relative_frobenius_error(self) -> float:
        """
        Compute the relative Frobenius norm error.
        
        ||H_approx - H_exact||_F / ||H_exact||_F
        
        Returns:
            Relative Frobenius norm error
        """
        exact_norm = torch.norm(self.exact_H, p='fro')
        if exact_norm == 0:
            return float('inf')
        return (torch.norm(self.error, p='fro') / exact_norm).item()
    
    def spectral_norm_error(self) -> float:
        """
        Compute the spectral norm (largest singular value) of the error matrix.
        
        ||H_approx - H_exact||_2
        
        Returns:
            Spectral norm of the error
        """
        return torch.norm(self.error, p=2).item()
    
    def relative_spectral_error(self) -> float:
        """
        Compute the relative spectral norm error.
        
        ||H_approx - H_exact||_2 / ||H_exact||_2
        
        Returns:
            Relative spectral norm error
        """
        exact_norm = torch.norm(self.exact_H, p=2)
        if exact_norm == 0:
            return float('inf')
        return (torch.norm(self.error, p=2) / exact_norm).item()
    
    def element_wise_mse(self) -> float:
        """
        Compute the mean squared error between elements.
        
        MSE = mean((H_approx - H_exact)^2)
        
        Returns:
            Mean squared error
        """
        return torch.mean(self.error ** 2).item()
    
    def full_hessian_mse(self) -> float:
        """
        Compute the full Hessian matrix MSE evaluation metric.
        
        Full Hessian MSE = (1/d^2) * sum_{i,j} (H_approx[i,j] - H_exact[i,j])^2
        
        This provides a comprehensive evaluation of the entire Hessian matrix approximation
        quality, normalized by the matrix size.
        
        Returns:
            Full Hessian MSE value (normalized by matrix size)
        """
        # Compute element-wise squared error
        squared_error = self.error ** 2
        
        # Sum all elements and normalize by total number of elements
        total_elements = self.exact_H.shape[0] * self.exact_H.shape[1]
        full_mse = torch.sum(squared_error).item() / total_elements
        
        return full_mse
    
    def relative_full_hessian_mse(self, eps: float = 1e-8) -> float:
        """
        Compute the relative full Hessian matrix MSE.
        
        Relative Full Hessian MSE = Full_MSE / (||H_exact||_F^2 / d^2 + eps)
        
        This normalizes the full Hessian MSE by the scale of the exact Hessian.
        
        Args:
            eps (float): Small value to prevent division by zero
            
        Returns:
            Relative full Hessian MSE value
        """
        # Compute full Hessian MSE
        full_mse = self.full_hessian_mse()
        
        # Compute normalized squared Frobenius norm of exact Hessian
        exact_frobenius_sq = torch.norm(self.exact_H, p='fro').item() ** 2
        total_elements = self.exact_H.shape[0] * self.exact_H.shape[1]
        normalized_exact_norm_sq = exact_frobenius_sq / total_elements
        
        # Avoid division by zero
        if normalized_exact_norm_sq < eps:
            return float('inf')
        
        # Compute relative MSE
        relative_full_mse = full_mse / (normalized_exact_norm_sq + eps)
        
        return relative_full_mse
    
    def element_wise_mae(self) -> float:
        """
        Compute the mean absolute error between elements.
        
        MAE = mean(|H_approx - H_exact|)
        
        Returns:
            Mean absolute error
        """
        return torch.mean(torch.abs(self.error)).item()
    
    def directional_curvature_error(self, num_directions: int = 100, eps: float = 1e-8) -> float:
        """
        Compute the directional curvature error (DCE).

        DCE measures how well the approximate Hessian preserves curvature
        along random directions, defined as:

            DCE = mean_v |vᵀ (H_approx - H_exact) v| / (|vᵀ H_exact v| + eps)

        Args:
            num_directions (int): Number of random directions to sample.
            eps (float): Small value to prevent division by zero.

        Returns:
            float: Mean directional curvature error.
        """
        n = self.exact_H.shape[0]
        errors = []

        for _ in range(num_directions):
            # Generate a random unit vector v
            v = torch.randn(n, device=self.exact_H.device)
            v = v / (torch.norm(v) + eps)

            # Compute curvature in this direction
            exact_curvature = v @ self.exact_H @ v
            approx_curvature = v @ self.approx_H @ v

            # Directional curvature error (relative)
            error = torch.abs(approx_curvature - exact_curvature) / (torch.abs(exact_curvature) + eps)
            errors.append(error.item())

        # Return mean directional curvature error
        return float(torch.tensor(errors).mean())
    
    def operator_norm_error(self, delta: float = 1e-8) -> float:
        """
        Compute the relative operator norm error.

        ε_rel = ||H_approx - H_exact||_2 / (max{||H_exact||_2, ||H_approx||_2} + δ)

        where ||·||_2 denotes the operator (spectral) norm, i.e., the largest singular value.

        Args:
            delta (float): Small constant for numerical stability.

        Returns:
            float: Relative operator norm error.
        """
        # Check if matrices contain non-finite values
        if not torch.all(torch.isfinite(self.approx_H)):
            print("Warning: approx_H contains non-finite values; returning large error value")
            return float('inf')
        
        if not torch.all(torch.isfinite(self.exact_H)):
            print("Warning: exact_H contains non-finite values; returning large error value")
            return float('inf')
        
        # Compute spectral (operator) norms using largest singular value
        try:
            error_norm = torch.linalg.norm(self.approx_H - self.exact_H, ord=2)
            exact_norm = torch.linalg.norm(self.exact_H, ord=2)
            approx_norm = torch.linalg.norm(self.approx_H, ord=2)

            denominator = torch.max(exact_norm, approx_norm) + delta
            rel_error = error_norm / denominator

            return rel_error.item()
        except Exception as e:
            print(f"Warning: error computing operator norm: {e}; returning large error value")
            return float('inf')

    def approximation_error(self, gradient_vector: Union[torch.Tensor, np.ndarray], eps: float = 1e-8) -> float:
        """
        Compute the approximation error metric as defined in the research paper.

        Approximation Error = (1/N) * Σ_{i=1}^{N} [ || H * Ĥ⁻¹ * v_i - v_i ||² / || v_i ||² ]

        For this implementation, N=1 and v_i is the unit gradient vector at the current test point.
        Both Hessian matrices are diagonalized (only diagonal elements kept) to avoid matrix inversion issues.

        Args:
            gradient_vector: The gradient vector at the current test point
            eps (float): Small value to prevent division by zero

        Returns:
            float: Approximation error value
        """
        # Convert to torch tensor if numpy array
        if isinstance(gradient_vector, np.ndarray):
            gradient_vector = torch.from_numpy(gradient_vector)
        
        v = gradient_vector.float()
        
        v_unit = v
        
        # Check for finite values
        if not torch.all(torch.isfinite(self.exact_H)) or not torch.all(torch.isfinite(self.approx_H)):
            print("Warning: Hessian matrices contain non-finite values; returning large error value")
            return float('inf')
        
        try:

            H_approx_H_inv_v = self.approx_H @ self.exact_H.inverse() @ v_unit
            # Compute the numerator: || H * Ĥ⁻¹ * v_unit - v_unit ||²
            numerator = torch.norm(H_approx_H_inv_v - v_unit) ** 2
            
            # Compute the denominator: || v_unit ||² (should be 1 for unit vector)
            denominator = torch.norm(v_unit) ** 2
            
            # Compute approximation error
            approx_error = numerator / (denominator + eps)
            
            return approx_error.item()
            
        except Exception as e:
            print(f"Warning: error computing approximation error: {e}; returning large error value")
            return float('inf')

    
    def diagonal_mse(self) -> float:
        """
        Compute the mean squared error of diagonal elements.
        
        MSE = mean((diag(H_approx) - diag(H_exact))^2)
        
        Returns:
            Mean squared error of diagonal elements
        """
        exact_diag = torch.diag(self.exact_H)
        approx_diag = torch.diag(self.approx_H)
        
        return torch.mean((approx_diag - exact_diag) ** 2).item()
    
    def get_all_metrics(self, include_eigenvalue: bool = True, 
                       top_k_eigenvalues: int = 10,
                       gradient_vector: Optional[Union[torch.Tensor, np.ndarray]] = None) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            include_eigenvalue: Whether to include eigenvalue-based metrics
            top_k_eigenvalues: Number of top eigenvalues to use for comparisons
            gradient_vector: Gradient vector for approximation error metric (optional)
            
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {
            'frobenius_norm_error': self.frobenius_norm_error(),
            'relative_frobenius_error': self.relative_frobenius_error(),
            'spectral_norm_error': self.spectral_norm_error(),
            'relative_spectral_error': self.relative_spectral_error(),
            'element_wise_mse': self.element_wise_mse(),
            'full_hessian_mse': self.full_hessian_mse(),
            'relative_full_hessian_mse': self.relative_full_hessian_mse(),
            'element_wise_mae': self.element_wise_mae(),
            'directional_curvature_error': self.directional_curvature_error(),
            'operator_norm_error': self.operator_norm_error(),
            'diagonal_mse': self.diagonal_mse(),
        }
        
        # Add approximation error if gradient vector is provided
        if gradient_vector is not None:
            metrics['approximation_error'] = self.approximation_error(gradient_vector)
        
        return metrics
    
    def print_summary(self, include_eigenvalue: bool = True, 
                     gradient_vector: Optional[Union[torch.Tensor, np.ndarray]] = None):
        """
        Print a formatted summary of all metrics.
        
        Args:
            include_eigenvalue: Whether to include eigenvalue-based metrics
            gradient_vector: Gradient vector for approximation error metric (optional)
        """
        metrics = self.get_all_metrics(include_eigenvalue=include_eigenvalue, 
                                      gradient_vector=gradient_vector)
        
        print("=" * 60)
        print("Hessian Approximation Evaluation Metrics")
        print("=" * 60)
        print(f"Matrix shape: {self.exact_H.shape}")
        print()
        
        print("Norm-based Metrics:")
        print("-" * 60)
        print(f"  Frobenius Norm Error:          {metrics['frobenius_norm_error']:.6e}")
        print(f"  Relative Frobenius Error:      {metrics['relative_frobenius_error']:.6f}")
        print()
        
        print("New Error Metrics:")
        print("-" * 60)
        print(f"  Directional Curvature Error:   {metrics['directional_curvature_error']:.6e}")
        print(f"  Operator Norm Error:           {metrics['operator_norm_error']:.6f}")
        print(f"  Diagonal MSE:                  {metrics['diagonal_mse']:.6e}")
        print(f"  Full Hessian MSE:              {metrics['full_hessian_mse']:.6e}")
        print(f"  Relative Full Hessian MSE:     {metrics['relative_full_hessian_mse']:.6f}")
        
        # Add approximation error if gradient vector was provided
        if gradient_vector is not None and 'approximation_error' in metrics:
            print(f"  Approximation Error:           {metrics['approximation_error']:.6e}")
        
        print()
        print("=" * 60)


def compare_hessians(exact_hessian: Union[torch.Tensor, np.ndarray],
                    approx_hessian: Union[torch.Tensor, np.ndarray],
                    print_summary: bool = True,
                    include_eigenvalue: bool = True,
                    gradient_vector: Optional[Union[torch.Tensor, np.ndarray]] = None) -> Dict[str, float]:
    """
    Convenience function to compare two Hessian matrices.
    
    Args:
        exact_hessian: The exact/true Hessian matrix
        approx_hessian: The approximated Hessian matrix
        print_summary: Whether to print a formatted summary
        include_eigenvalue: Whether to include eigenvalue-based metrics
        gradient_vector: Gradient vector for approximation error metric (optional)
        
    Returns:
        Dictionary containing all computed metrics
    """
    evaluator = HessianMetrics(exact_hessian, approx_hessian)
    
    if print_summary:
        evaluator.print_summary(include_eigenvalue=include_eigenvalue, 
                               gradient_vector=gradient_vector)
    
    return evaluator.get_all_metrics(include_eigenvalue=include_eigenvalue, 
                                    gradient_vector=gradient_vector)
