import numpy as np


def hessian_CD_approx(theta, test_function, d, K, mu, noise_std, sample_randomseed):
    K = int((K-1)/2)
    np.random.seed(sample_randomseed)  
    U = np.random.randn(d, K)  # Each column is one direction u_k
    
    xi = np.random.randn(d) * noise_std
    f_x = test_function.f(theta, xi)
    H_approx = np.zeros((d, d))
    
    for k in range(K):
        u_k = U[:, k]

        theta_plus = theta + mu * u_k
        f_plus = test_function.f(theta_plus, xi)

        theta_minus = theta - mu * u_k
        f_minus = test_function.f(theta_minus, xi)
        
        second_diff = (f_plus - 2 * f_x + f_minus) / (mu ** 2)
        u_k_col = u_k[:, np.newaxis]  # Convert to column vector
        H_approx += second_diff * (u_k_col @ u_k_col.T)
    
    # Average over symmetric finite differences
    H_approx = H_approx / (2*K)

    
    return H_approx

def hessian_SCD_approx(theta, test_function, d, K, mu, noise_std, sample_randomseed):
    K = int((K-1)/2)
    np.random.seed(sample_randomseed)  # For reproducibility
    U = np.random.randn(d, K)  # Each column is one direction u_k
    

    xi = np.random.randn(d) * noise_std
    
    f_x = test_function.f(theta, xi)

    H_approx = np.zeros((d, d))
    
    for k in range(K):
        u_k = U[:, k]
        
        theta_plus = theta + mu * u_k
        f_plus = test_function.f(theta_plus, xi)
        
        theta_minus = theta - mu * u_k
        f_minus = test_function.f(theta_minus, xi)

        second_diff = (f_plus - 2 * f_x + f_minus) / (mu ** 2)
        
        u_k_col = u_k[:, np.newaxis]  # Convert to column vector
        I = np.eye(d, dtype=np.float32)

        H_approx += second_diff * (u_k_col @ u_k_col.T - I)

    H_approx = H_approx / (2*K)
    
    return H_approx

def hessian_GS_zeroBaseline_approx(theta, test_function, d, K, mu, noise_std, sample_randomseed):
    np.random.seed(sample_randomseed)
    U= np.random.randn(d, K)                    # (d,)
    xi = np.random.randn(d) * noise_std
    H_approx = np.zeros((d, d))
    for k in range(K):
        u_k = U[:, k]
        theta_perturbed = theta + mu * u_k          # (d,)
        f_val = test_function.f(theta_perturbed, xi)
        u_col = u_k[:, np.newaxis]
        I = np.eye(d, dtype=np.float32)
        H_approx += (f_val / mu**2) * (u_col @ u_col.T - I)
    return H_approx / K

def hessian_GS_originBaseline_approx(theta, test_function, d, K, mu, noise_std, sample_randomseed):
    
    np.random.seed(sample_randomseed)
    U = np.random.randn(d, K)  # Each column is a u_k

    xi = np.random.randn(d) * noise_std
    # Calculate f(theta + mu*u_k; xi) for all k
    f_vals = np.zeros(K)
    for i in range(K):
        u_k = U[:, i]
        theta_perturbed = theta + mu * u_k
        f_vals[i] = test_function.f(theta_perturbed, xi)

    # Calculate the average function value f_bar
    f_xi = test_function.f(theta, xi)

    # Calculate the Hessian approximation
    H_approx = np.zeros((d, d))
    for i in range(K):
        u_k = U[:, i]
        u_k_col = u_k[:, np.newaxis]  # For outer product
        term = (f_vals[i] - f_xi) / (mu**2)
        I = np.eye(d, dtype=np.float32)
        H_approx += term * (u_k_col @ u_k_col.T - I)

    return H_approx / K



def hessian_EGS_meanBaseline_approx(theta, test_function, d, K, mu, noise_std, sample_randomseed):
    """
    Implements the first Hessian approximation formula.
    H1 = (1/K) * sum_k { [f(theta+mu*u_k; xi) - f_bar] / mu^2 } * u_k*u_k.T
    where f_bar = (1/K) * sum_j f(theta+mu*u_j; xi)
    """
    # Generate K random direction vectors u_k from N(0, I)
    np.random.seed(sample_randomseed)
    U = np.random.randn(d, K)  # Each column is a u_k
    xi = np.random.randn(d) * noise_std

    # Calculate f(theta + mu*u_k; xi) for all k
    f_vals = np.zeros(K)
    for i in range(K):
        u_k = U[:, i]
        theta_perturbed = theta + mu * u_k
        f_vals[i] = test_function.f(theta_perturbed, xi)

    # Calculate the average function value f_bar
    f_bar = np.mean(f_vals)
    # Calculate the Hessian approximation
    H_approx = np.zeros((d, d))
    for i in range(K):
        u_k = U[:, i]
        u_k_col = u_k[:, np.newaxis]  # For outer product
        term = (f_vals[i] - f_bar) / (mu**2)
        H_approx += term * (u_k_col @ u_k_col.T)

    return H_approx / (K - 1)

def hessian_EGS_meanBaseline_reuse_approx(theta, test_function, d, K, mu, noise_std, history_values, sample_randomseed):
    """
    Implements the first Hessian approximation formula with historical information reuse.
    
    Based on the formula in the image:
    H_hat(theta_{t-1}) = (1/(N*K)) * sum_{n,k=1}^{N,K} (1/mu^2) * [(u_{t-n,k} * u_{t-n,k}^T - I_d) * f(theta_{t-n} + mu * u_{t-n,k}; xi)]
    
    This uses N rounds of data (current round + N-1 previous rounds) with K points from each round.
    The method internally generates perturbations for all rounds and reuses historical test samples.
    If no historical information is available, falls back to the non-reuse implementation.
    
    Args:
        theta: Current parameter point (d-dimensional vector)
        test_function: Test function object
        d: Dimension
        K: Number of points per round
        mu: Perturbation parameter
        noise_std: Noise standard deviation
        history_values: Dictionary containing historical data with keys:
            - 'thetas': List of historical parameter vectors [theta_{t-1}, theta_{t-2}, ..., theta_{t-N+1}]
            - 'N': Number of rounds to use (including current)
    """
    # Extract historical information
    historical_thetas = history_values.get('thetas', [])
    N = 1 + len(historical_thetas)
    randomseeds = history_values.get('historical_randomseeds',[])
    # Check if we have any historical information
    if N <= 1 or len(historical_thetas) == 0:
        # No historical information available, use non-reuse implementation
        return hessian_EGS_meanBaseline_approx(theta, test_function, d, K, mu, noise_std, sample_randomseed)
    
    # Prepare all rounds: current + historical
    all_thetas = [theta] + historical_thetas[:N-1]  # Current round first
    all_randomseed = [sample_randomseed] + randomseeds[:N-1]
    
    # Generate perturbations for all rounds with corresponding random seeds
    all_perturbations = []
    for n in range(N):
        # Use the random seed corresponding to this round
        np.random.seed(all_randomseed[n])
        U_n = np.random.randn(d, K)
        all_perturbations.append(U_n)
    
    # Generate a single random vector xi for this estimation run (Common Random Numbers)
    xi = np.random.randn(d) * noise_std
    
    # Calculate all function values for the mean computation
    all_f_vals = []
    for n in range(N):
        theta_n = all_thetas[n]
        U_n = all_perturbations[n]
        
        for k in range(K):
            u_nk = U_n[:, k]
            theta_perturbed = theta_n + mu * u_nk
            f_perturbed = test_function.f(theta_perturbed, xi)
            all_f_vals.append(f_perturbed)
    
    # Calculate the global mean f_bar over all N*K points
    f_bar = np.mean(all_f_vals)
    
    # Calculate the Hessian approximation using the formula
    H_approx = np.zeros((d, d))
    total_terms = 0
    
    for n in range(N):
        theta_n = all_thetas[n]
        U_n = all_perturbations[n]
        
        for k in range(K):
            u_nk = U_n[:, k]
            u_nk_col = u_nk[:, np.newaxis]
            term = (all_f_vals[n*K + k] - f_bar) / (mu**2)
            H_approx += term * (u_nk_col @ u_nk_col.T)
            total_terms += 1
    
    # Average over all N*K terms
    return H_approx / (total_terms - 1)
