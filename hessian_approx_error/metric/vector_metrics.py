"""
Vector Approximation Evaluation Metrics

This module provides various metrics to evaluate the quality of vector approximations
by comparing them with the exact vector.
"""

import torch
import numpy as np
from typing import Dict, Optional, Union


class VectorMetrics:
    """
    A class to compute various metrics for evaluating vector approximations.
    """
    
    def __init__(self, exact_vector: Union[torch.Tensor, np.ndarray], 
                 approx_vector: Union[torch.Tensor, np.ndarray]):
        """
        Initialize with exact and approximate vectors.
        
        Args:
            exact_vector: The exact/true vector
            approx_vector: The approximated vector
        """
        # Convert to torch tensors if numpy arrays
        if isinstance(exact_vector, np.ndarray):
            exact_vector = torch.from_numpy(exact_vector)
        if isinstance(approx_vector, np.ndarray):
            approx_vector = torch.from_numpy(approx_vector)
            
        self.exact_v = exact_vector.float()
        self.approx_v = approx_vector.float()
        
        # Verify dimensions match
        assert self.exact_v.shape == self.approx_v.shape, \
            f"Shape mismatch: exact {self.exact_v.shape} vs approx {self.approx_v.shape}"
        
        # Compute error vector once
        self.error = self.approx_v - self.exact_v
        
    def mse(self) -> float:
        """
        Compute the mean squared error.
        
        MSE = (1/d) * sum_i (v_approx[i] - v_exact[i])^2
        
        Returns:
            Mean squared error
        """
        return torch.mean(self.error ** 2).item()
    
    def rmse(self) -> float:
        """
        Compute the root mean squared error.
        
        RMSE = sqrt(MSE)
        
        Returns:
            Root mean squared error
        """
        return torch.sqrt(torch.mean(self.error ** 2)).item()
    
    def mae(self) -> float:
        """
        Compute the mean absolute error.
        
        MAE = (1/d) * sum_i |v_approx[i] - v_exact[i]|
        
        Returns:
            Mean absolute error
        """
        return torch.mean(torch.abs(self.error)).item()
    
    def relative_mse(self, eps: float = 1e-8) -> float:
        """
        Compute the relative mean squared error.
        
        Relative MSE = MSE / (mean(v_exact^2) + eps)
        
        Args:
            eps: Small value to prevent division by zero
            
        Returns:
            Relative mean squared error
        """
        exact_mean_sq = torch.mean(self.exact_v ** 2).item()
        if exact_mean_sq < eps:
            return float('inf')
        return self.mse() / (exact_mean_sq + eps)
    
    def relative_mae(self, eps: float = 1e-8) -> float:
        """
        Compute the relative mean absolute error.
        
        Relative MAE = MAE / (mean(|v_exact|) + eps)
        
        Args:
            eps: Small value to prevent division by zero
            
        Returns:
            Relative mean absolute error
        """
        exact_mean_abs = torch.mean(torch.abs(self.exact_v)).item()
        if exact_mean_abs < eps:
            return float('inf')
        return self.mae() / (exact_mean_abs + eps)
    
    def max_absolute_error(self) -> float:
        """
        Compute the maximum absolute error.
        
        Max AE = max_i |v_approx[i] - v_exact[i]|
        
        Returns:
            Maximum absolute error
        """
        return torch.max(torch.abs(self.error)).item()
    
    def relative_max_error(self, eps: float = 1e-8) -> float:
        """
        Compute the relative maximum error.
        
        Relative Max Error = Max AE / (max(|v_exact|) + eps)
        
        Args:
            eps: Small value to prevent division by zero
            
        Returns:
            Relative maximum error
        """
        exact_max = torch.max(torch.abs(self.exact_v)).item()
        if exact_max < eps:
            return float('inf')
        return self.max_absolute_error() / (exact_max + eps)
    
    def l2_norm_error(self) -> float:
        """
        Compute the L2 norm of the error vector.
        
        L2 Error = ||v_approx - v_exact||_2
        
        Returns:
            L2 norm of the error
        """
        return torch.norm(self.error, p=2).item()
    
    def relative_l2_error(self, eps: float = 1e-8) -> float:
        """
        Compute the relative L2 norm error.
        
        Relative L2 Error = ||v_approx - v_exact||_2 / (||v_exact||_2 + eps)
        
        Args:
            eps: Small value to prevent division by zero
            
        Returns:
            Relative L2 norm error
        """
        exact_norm = torch.norm(self.exact_v, p=2).item()
        if exact_norm < eps:
            return float('inf')
        return self.l2_norm_error() / (exact_norm + eps)
    
    def cosine_similarity(self, eps: float = 1e-8) -> float:
        """
        Compute the cosine similarity between vectors.
        
        Cosine Similarity = (v_exact · v_approx) / (||v_exact||_2 * ||v_approx||_2)
        
        Returns:
            Cosine similarity (1 = perfect match, -1 = opposite)
        """
        exact_norm = torch.norm(self.exact_v, p=2)
        approx_norm = torch.norm(self.approx_v, p=2)
        
        if exact_norm < eps or approx_norm < eps:
            return 0.0
        
        dot_product = torch.dot(self.exact_v, self.approx_v)
        return (dot_product / (exact_norm * approx_norm + eps)).item()
    
    def cosine_distance(self) -> float:
        """
        Compute the cosine distance between vectors.
        
        Cosine Distance = 1 - Cosine Similarity
        
        Returns:
            Cosine distance (0 = perfect match, 2 = opposite)
        """
        return 1.0 - self.cosine_similarity()
    
    def pearson_correlation(self, eps: float = 1e-8) -> float:
        """
        Compute the Pearson correlation coefficient.
        
        Returns:
            Pearson correlation coefficient (1 = perfect positive correlation)
        """
        # Center the vectors
        exact_centered = self.exact_v - torch.mean(self.exact_v)
        approx_centered = self.approx_v - torch.mean(self.approx_v)
        
        # Compute correlation
        numerator = torch.sum(exact_centered * approx_centered)
        denominator = torch.sqrt(torch.sum(exact_centered ** 2) * torch.sum(approx_centered ** 2))
        
        if denominator < eps:
            return 0.0
        
        return (numerator / (denominator + eps)).item()
    
    def normalized_mse(self, eps: float = 1e-8) -> float:
        """
        Compute the normalized mean squared error.
        
        Normalized MSE = MSE / var(v_exact)
        
        Args:
            eps: Small value to prevent division by zero
            
        Returns:
            Normalized mean squared error
        """
        exact_var = torch.var(self.exact_v).item()
        if exact_var < eps:
            return float('inf')
        return self.mse() / (exact_var + eps)
    
    def element_wise_relative_error(self, eps: float = 1e-8) -> float:
        """
        Compute the mean element-wise relative error.
        
        Element-wise Relative Error = mean_i |v_approx[i] - v_exact[i]| / (|v_exact[i]| + eps)
        
        Args:
            eps: Small value to prevent division by zero
            
        Returns:
            Mean element-wise relative error
        """
        relative_errors = torch.abs(self.error) / (torch.abs(self.exact_v) + eps)
        return torch.mean(relative_errors).item()
    
    def r_squared(self) -> float:
        """
        Compute the R-squared (coefficient of determination).
        
        R^2 = 1 - (SS_res / SS_tot)
        
        where SS_res = sum((v_exact - v_approx)^2)
              SS_tot = sum((v_exact - mean(v_exact))^2)
        
        Returns:
            R-squared value (1 = perfect fit, 0 = no better than mean)
        """
        ss_res = torch.sum(self.error ** 2)
        ss_tot = torch.sum((self.exact_v - torch.mean(self.exact_v)) ** 2)
        
        if ss_tot < 1e-8:
            return float('nan')
        
        return (1 - ss_res / ss_tot).item()
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {
            'mse': self.mse(),
            'rmse': self.rmse(),
            'mae': self.mae(),
            'relative_mse': self.relative_mse(),
            'relative_mae': self.relative_mae(),
            'max_absolute_error': self.max_absolute_error(),
            'relative_max_error': self.relative_max_error(),
            'l2_norm_error': self.l2_norm_error(),
            'relative_l2_error': self.relative_l2_error(),
            'cosine_similarity': self.cosine_similarity(),
            'cosine_distance': self.cosine_distance(),
            'pearson_correlation': self.pearson_correlation(),
            'normalized_mse': self.normalized_mse(),
            'element_wise_relative_error': self.element_wise_relative_error(),
            'r_squared': self.r_squared(),
        }
        
        return metrics
    
    def print_summary(self):
        """
        Print a formatted summary of all metrics.
        """
        metrics = self.get_all_metrics()
        
        print("=" * 60)
        print("Vector Approximation Evaluation Metrics")
        print("=" * 60)
        print(f"Vector dimension: {self.exact_v.shape[0]}")
        print()
        
        print("Error-based Metrics:")
        print("-" * 60)
        print(f"  MSE:                           {metrics['mse']:.6e}")
        print(f"  RMSE:                          {metrics['rmse']:.6e}")
        print(f"  MAE:                           {metrics['mae']:.6e}")
        print(f"  Max Absolute Error:            {metrics['max_absolute_error']:.6e}")
        print()
        
        print("Relative Error Metrics:")
        print("-" * 60)
        print(f"  Relative MSE:                  {metrics['relative_mse']:.6f}")
        print(f"  Relative MAE:                  {metrics['relative_mae']:.6f}")
        print(f"  Relative Max Error:            {metrics['relative_max_error']:.6f}")
        print(f"  Element-wise Relative Error:   {metrics['element_wise_relative_error']:.6f}")
        print()
        
        print("Norm-based Metrics:")
        print("-" * 60)
        print(f"  L2 Norm Error:                 {metrics['l2_norm_error']:.6e}")
        print(f"  Relative L2 Error:             {metrics['relative_l2_error']:.6f}")
        print()
        
        print("Similarity Metrics:")
        print("-" * 60)
        print(f"  Cosine Similarity:             {metrics['cosine_similarity']:.6f}")
        print(f"  Cosine Distance:               {metrics['cosine_distance']:.6f}")
        print(f"  Pearson Correlation:           {metrics['pearson_correlation']:.6f}")
        print(f"  R-squared:                     {metrics['r_squared']:.6f}")
        print()
        
        print("Normalized Metrics:")
        print("-" * 60)
        print(f"  Normalized MSE:                {metrics['normalized_mse']:.6f}")
        print()
        print("=" * 60)


def compare_vectors(exact_vector: Union[torch.Tensor, np.ndarray],
                   approx_vector: Union[torch.Tensor, np.ndarray],
                   print_summary: bool = True) -> Dict[str, float]:
    """
    Convenience function to compare two vectors.
    
    Args:
        exact_vector: The exact/true vector
        approx_vector: The approximated vector
        print_summary: Whether to print a formatted summary
        
    Returns:
        Dictionary containing all computed metrics
    """
    evaluator = VectorMetrics(exact_vector, approx_vector)
    
    if print_summary:
        evaluator.print_summary()
    
    return evaluator.get_all_metrics()

