"""
Test function collection with analytic Hessians

Provides multiple test functions with known Hessian matrices to evaluate the
accuracy of Hessian approximation methods.
"""

import numpy as np
from typing import Tuple, Optional


class TestFunction:
    """
    Base class for test functions.
    """
    def __init__(self, d: int, seed: int = 27):
        """
        Args:
            d: Dimension
            seed: Random seed
        """
        self.d = d
        self.seed = seed
        np.random.seed(seed)
        
    def f(self, x: np.ndarray, xi: Optional[np.ndarray] = None) -> float:
        """
        Compute function value
        
        Args:
            x: Input point
            xi: Random noise (optional)
        Returns:
            Function value
        """
        raise NotImplementedError
    
    def grad_f(self, x: np.ndarray, xi: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gradient
        
        Args:
            x: Input point
            xi: Random noise (optional)
        Returns:
            Gradient vector
        """
        raise NotImplementedError
    
    def hessian_f(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the exact Hessian matrix (analytic)
        
        Args:
            x: Input point
        Returns:
            Hessian matrix
        """
        raise NotImplementedError
        
    def inv_hessian_f(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the exact inverse Hessian matrix (analytic)
        
        Args:
            x: Input point
        Returns:
            Inverse Hessian matrix
        """
        # Default through numerical computation of Hessian's inverse
        H = self.hessian_f(x)
        # Add small regularization term for numerical stability
        reg_term = 1e-10 * np.trace(H) / self.d
        H_reg = H + reg_term * np.eye(self.d)
        return np.linalg.inv(H_reg)
    
    def hessian_vector_product(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute Hessian-vector product (H·v)
        
        Args:
            x: Input point
            v: Vector
        Returns:
            Hessian-vector product
        """
        # Default through computation of Hessian matrix then multiplication
        H = self.hessian_f(x)
        return H @ v
    
    def inv_hessian_vector_product(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute inverse Hessian-vector product (H^{-1}·v)
        
        Args:
            x: Input point
            v: Vector
        Returns:
            Inverse Hessian-vector product
        """
        # Default through computation of Hessian inverse then multiplication
        H_inv = self.inv_hessian_f(x)
        return H_inv @ v

class QuadraticFunction(TestFunction):
    """
    Quadratic function: f(x) = 0.5 * x^T A x + b^T x + c
    
    Hessian matrix: H = A (constant with respect to x)
    """
    def __init__(self, d: int, seed: int = 27, condition_number: float = 100.0):
        """
        Args:
            d: Dimension
            seed: Random seed
            condition_number: Condition number (controls the conditioning of A)
        """
        super().__init__(d, seed)
        
        # Generate symmetric positive definite matrix A
        np.random.seed(seed)
        # Construct matrix with specified condition number through eigenvalue decomposition
        U = np.linalg.qr(np.random.randn(d, d))[0]  # Orthogonal matrix
        
        # Construct eigenvalues: uniformly distributed from 1 to condition_number
        eigenvalues = np.linspace(1, condition_number, d)
        Lambda = np.diag(eigenvalues)
        
        # A = U * Lambda * U^T
        self.A = U @ Lambda @ U.T
        self.A = (self.A + self.A.T) / 2  # Ensure symmetry
        
        # Randomly generate b and c
        self.b = np.random.randn(d)
        self.c = np.random.randn()
        
        # Precompute Hessian's inverse (for quadratic functions, Hessian is constant)
        self.A_inv = None  # Lazy initialization
        
        print(f"Quadratic function initialized:")
        print(f"  Dimension: {d}")
        print(f"  Condition number: {np.linalg.cond(self.A):.2f}")
        print(f"  Minimum eigenvalue: {np.linalg.eigvals(self.A).min():.4f}")
        print(f"  Maximum eigenvalue: {np.linalg.eigvals(self.A).max():.4f}")
        
    def f(self, x: np.ndarray, xi: Optional[np.ndarray] = None) -> float:
        """Compute function value"""
        base_value = 0.5 * x.T @ self.A @ x + self.b.T @ x + self.c
        
        # Add random noise (if provided)
        if xi is not None:
            return base_value + xi.T @ x
        return base_value
    
    def grad_f(self, x: np.ndarray, xi: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute gradient"""
        base_grad = self.A @ x + self.b
        
        # Add random noise (if provided)
        if xi is not None:
            return base_grad + xi
        return base_grad
    
    def hessian_f(self, x: np.ndarray) -> np.ndarray:
        """
        Return the true Hessian matrix
        For a quadratic function, the Hessian is A (constant with respect to x)
        """
        return self.A.copy()
    
    def inv_hessian_f(self, x: np.ndarray) -> np.ndarray:
        """
        Return the true inverse Hessian
        For a quadratic function, the inverse is A^(-1) (constant with respect to x)
        """
        # Lazy initialization, compute inverse only once
        if self.A_inv is None:
            # Add small regularization term for numerical stability
            reg_term = 1e-10 * np.trace(self.A) / self.d
            A_reg = self.A + reg_term * np.eye(self.d)
            self.A_inv = np.linalg.inv(A_reg)
        return self.A_inv.copy()
    
    def hessian_vector_product(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute Hessian-vector product (A·v)
        For a quadratic function, use matrix multiplication
        """
        return self.A @ v
    
    def inv_hessian_vector_product(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute inverse Hessian-vector product (A^(-1)·v)
        """
        return self.inv_hessian_f(x) @ v

class RosenbrockFunction(TestFunction):
    """
    N-dimensional Rosenbrock function
    
    f(x) = Σ_{i=1}^{d-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    
    A non-convex function commonly used as a test function in optimization
    """
    def __init__(self, d: int, seed: int = 27, alpha: float =10.0):
        """
        Args:
            d: Dimension (at least 2)
            seed: Random seed
            alpha: Rosenbrock parameter (usually 100)
        """
        super().__init__(d, seed)
        assert d >= 2, "Rosenbrock function requires at least 2 dimensions"
        self.alpha = alpha
        
        print(f"Rosenbrock function initialized:")
        print(f"  Dimension: {d}")
        print(f"  Parameter α: {alpha}")
        print(f"  Global minimum point: x = [1, 1, ..., 1]")
        print(f"  Minimum value: f(x*) = 0")
        
    def f(self, x: np.ndarray, xi: Optional[np.ndarray] = None) -> float:
        """Compute function value"""
        value = 0.0
        for i in range(self.d - 1):
            value += self.alpha * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        
        # Add random noise (if provided)
        if xi is not None:
            value += xi.T @ x
        
        return value
    
    def grad_f(self, x: np.ndarray, xi: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute gradient"""
        grad = np.zeros(self.d)
        
        # First component
        grad[0] = -4 * self.alpha * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        
        # Middle components
        for i in range(1, self.d - 1):
            grad[i] = 2 * self.alpha * (x[i] - x[i-1]**2) - \
                      4 * self.alpha * x[i] * (x[i+1] - x[i]**2) - \
                      2 * (1 - x[i])
        
        # Last component
        grad[self.d - 1] = 2 * self.alpha * (x[self.d-1] - x[self.d-2]**2)
        
        # Add random noise (if provided)
        if xi is not None:
            grad += xi
        
        return grad
    
    def hessian_f(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the true Hessian matrix
        Rosenbrock's Hessian is primarily tridiagonal
        """
        H = np.zeros((self.d, self.d))
        
        # Diagonal elements
        for i in range(self.d):
            if i == 0:
                H[i, i] = -4 * self.alpha * (x[1] - 3*x[0]**2) + 2
            elif i == self.d - 1:
                H[i, i] = 2 * self.alpha
            else:
                H[i, i] = 2 * self.alpha - 4 * self.alpha * (x[i+1] - 3*x[i]**2) + 2
        
        # Off-diagonal elements (only adjacent elements are non-zero)
        for i in range(self.d - 1):
            H[i, i+1] = -4 * self.alpha * x[i]
            H[i+1, i] = -4 * self.alpha * x[i]  # Symmetry
        
        return H
        
    def inv_hessian_f(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of the Hessian for Rosenbrock
        """
        H = self.hessian_f(x)
        # Add small regularization term for numerical stability
        reg_term = 1e-10 * np.trace(H) / self.d
        H_reg = H + reg_term * np.eye(self.d)
        return np.linalg.inv(H_reg)
    
    def hessian_vector_product(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute Hessian-vector product (H·v)
        For Rosenbrock, leverage the sparse structure
        """
        result = np.zeros_like(v)
        
        # First component
        result[0] = (-4 * self.alpha * (x[1] - 3*x[0]**2) + 2) * v[0] + (-4 * self.alpha * x[0]) * v[1]
        
        # Middle components
        for i in range(1, self.d - 1):
            result[i] = (-4 * self.alpha * x[i-1]) * v[i-1] + \
                        (2 * self.alpha - 4 * self.alpha * (x[i+1] - 3*x[i]**2) + 2) * v[i] + \
                        (-4 * self.alpha * x[i]) * v[i+1]
        
        # Last component
        result[self.d-1] = (-4 * self.alpha * x[self.d-2]) * v[self.d-2] + (2 * self.alpha) * v[self.d-1]
        
        return result
    
    def inv_hessian_vector_product(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute inverse Hessian-vector product (H^(-1)·v)
        For Rosenbrock, solve the linear system H·x = v
        """
        H = self.hessian_f(x)
        # Add small regularization term for numerical stability
        reg_term = 1e-10 * np.trace(H) / self.d
        H_reg = H + reg_term * np.eye(self.d)
        
        # Solve linear system H·x = v
        return np.linalg.solve(H_reg, v)

class StyblinskiTangFunction(TestFunction):
    """
    Styblinski-Tang function (multimodal test function)
    
    f(x) = 0.5 * Σ_{i=1}^d (x_i^4 - 16*x_i^2 + 5*x_i)
    
    Global minimum: x_i ≈ -2.903534 for all i; f(x*) ≈ -39.16599 * d
    """
    def __init__(self, d: int, seed: int = 27):
        super().__init__(d, seed)
        
        print(f"Styblinski-Tang function initialized:")
        print(f"  Dimension: {d}")
        print(f"  Global minimum point: x_i ≈ -2.903534")
        print(f"  Minimum value: f(x*) ≈ {-39.16599 * d:.2f}")
        
    def f(self, x: np.ndarray, xi: Optional[np.ndarray] = None) -> float:
        """Compute function value"""
        value = 0.5 * np.sum(x**4 - 16*x**2 + 5*x)
        
        if xi is not None:
            value += xi.T @ x
        
        return value
    
    def grad_f(self, x: np.ndarray, xi: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute gradient"""
        grad = 0.5 * (4*x**3 - 32*x + 5)
        
        if xi is not None:
            grad += xi
        
        return grad
    
    def hessian_f(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Hessian
        For this function, the Hessian is diagonal with
        H_{ii} = 0.5 * (12*x_i^2 - 32)
        """
        diag_elements = 0.5 * (12*x**2 - 32)
        return np.diag(diag_elements)
        
    def inv_hessian_f(self, x: np.ndarray) -> np.ndarray:
        """
        Compute inverse Hessian
        For Styblinski-Tang, the Hessian is diagonal so inversion is element-wise
        """
        diag_elements = 0.5 * (12*x**2 - 32)
        
        # Add small regularization term for numerical stability
        reg_term = 1e-10 * np.abs(np.mean(diag_elements))
        diag_elements_reg = diag_elements + reg_term * np.sign(diag_elements)
        
        # Compute inverse diagonal elements
        inv_diag_elements = 1.0 / diag_elements_reg
        
        return np.diag(inv_diag_elements)
    
    def hessian_vector_product(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute Hessian-vector product (H·v)
        For this function, Hv is element-wise multiplication
        """
        diag_elements = 0.5 * (12*x**2 - 32)
        return diag_elements * v
    
    def inv_hessian_vector_product(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute inverse Hessian-vector product (H^(-1)·v)
        For this function, also element-wise
        """
        diag_elements = 0.5 * (12*x**2 - 32)
        
        # Add small regularization term for numerical stability
        reg_term = 1e-10 * np.abs(np.mean(diag_elements))
        diag_elements_reg = diag_elements + reg_term * np.sign(diag_elements)
        
        return v / diag_elements_reg

def create_test_function(function_type: str, d: int, **kwargs) -> TestFunction:
    """
    Factory method to create a test function.
    
    Args:
        function_type: Function type ('quadratic', 'rosenbrock', 'logistic', 'styblinski', 'exponential_quadratic')
        d: Dimension
        **kwargs: Additional parameters
    
    Returns:
        TestFunction instance
    """
    function_types = {
        'quadratic': QuadraticFunction,
        'rosenbrock': RosenbrockFunction,
        'styblinski': StyblinskiTangFunction,
    }
    
    if function_type not in function_types:
        raise ValueError(f"Unknown function type: {function_type}. "
                        f"Choices: {list(function_types.keys())}")
    
    return function_types[function_type](d, **kwargs)
