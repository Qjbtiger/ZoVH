#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metric'))


from test_functions import (
    create_test_function, 
    QuadraticFunction, 
    RosenbrockFunction, 
    StyblinskiTangFunction
)
from hessian_approx import (
    hessian_CD_approx,
    hessian_S1_approx,
    hessian_S2_approx,
    hessian_S3_approx,
    hessian_ZoVH_approx,
    hessian_ZoVH_reuse_approx
)


class HessianApproximationSimulator:
    """
    Hessian Approximation Simulator
    """
    
    def __init__(self, function_type: str, d: int, n_samples: int = 100, 
                 noise_std: float = 0.01, seed: int = 27):
        """
        Initialize the simulator
        
        Args:
            function_type: Test function type
            d: Dimension
            n_samples: Number of samples
            noise_std: Noise standard deviation
            seed: Random seed
        """
        self.function_type = function_type
        self.d = d
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.seed = seed
        
        np.random.seed(seed)
        self.test_function = self._create_test_function()
        self.results = {}
        self.samples = []
        self.sample_values = []
        
        print(f"Hessian Approximation Simulator initialized:")
        print(f"  Function type: {function_type}")
        print(f"  Dimension: {d}")
        print(f"  Number of samples: {n_samples}")
        print(f"  Noise std: {noise_std}")
        print()
    
    def _create_test_function(self):
        """Create test function"""
        if self.function_type == 'quadratic':
            return QuadraticFunction(self.d, seed=self.seed, condition_number=10.0)
        elif self.function_type == 'rosenbrock':
            return RosenbrockFunction(self.d, seed=self.seed)
        elif self.function_type == 'styblinski':
            return StyblinskiTangFunction(self.d, seed=self.seed)
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
    
    def sample_points(self, center_point: Optional[np.ndarray] = None) -> List[np.ndarray]:

        print(f"Sampling {self.n_samples} points...")
        
        if center_point is None:
            center_point = np.random.randn(self.d)
                

        samples = []
        
        xi = np.zeros(self.d)
        


        for i in range(self.n_samples - 1):
            # Compute current point's gradient
            gradient = self.test_function.grad_f(center_point, xi)
            
            # Check if gradient contains non-finite values
            if not np.all(np.isfinite(gradient)):
                print(f"Warning: Non-finite gradient at step {i+1}, stopping sampling")
                break
            
            # Normalize gradient to unit vector, then take a small step
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 1e-12:  # Avoid division by zero
                unit_gradient = gradient / gradient_norm
                step_size = 0.0001 # Small step size
                center_point = center_point - step_size * unit_gradient
                samples.append(center_point)
            # Check if parameters contain non-finite values
            if not np.all(np.isfinite(center_point)):
                print(f"Warning: Non-finite parameters at step {i+1}, stopping sampling")
                break
    
        # Sort by function value in descending order (from worst point to best point)

        
        print(f"Sampling completed")
        
        return samples

    def analyze_results(self) -> pd.DataFrame:
        """Analyze results and generate summary table"""
        print("\n" + "=" * 80)
        print("Results Analysis")
        print("=" * 80)
        
        # Create result summary table
        summary_data = []
        
        for method, results in self.results.items():
            if 'error' in results:
                continue
            
            errors = results['errors']
            metrics_list = results['metrics']
            
            # Check if there is valid error data
            if len(errors) == 0:
                print(f"Method {method} has no valid error data, skipping statistics")
                continue
            
            # Compute statistics
            mean_error = np.mean(errors)

            
            # Compute average metrics (including new approximation error and full Hessian MSE)
            selected_metric_names = [
                'mean_frobenius_error'
            ]
            
            avg_metrics = {}
            for metric_name in selected_metric_names:
                values = []
                for metrics in metrics_list:
                    if metric_name in metrics and not np.isnan(metrics[metric_name]):
                        values.append(metrics[metric_name])
                
                if values:
                    avg_metrics[f'avg_{metric_name}'] = np.mean(values)
                    # avg_metrics[f'std_{metric_name}'] = np.std(values)
                else:
                    avg_metrics[f'avg_{metric_name}'] = float('nan')
                    avg_metrics[f'std_{metric_name}'] = float('nan')
            
            summary_data.append({
                'Method': method,
                'Mean_Frobenius_Error': mean_error,
                **avg_metrics
            })
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Sort by mean Frobenius error
        df = df.sort_values('Mean_Frobenius_Error')
        
        print("\nMethod ranking (by mean Frobenius error):")
        print("-" * 80)
        print(df[['Method', 'Mean_Frobenius_Error', 'Std_Frobenius_Error']].to_string(index=False, float_format='%.4f'))
        
        # Display detailed metrics comparison
        self._print_detailed_metrics_comparison(df)
        
        return df
    
    def _print_detailed_metrics_comparison(self, df: pd.DataFrame):
        """Print detailed metrics comparison"""
        print("\n" + "=" * 80)
        print("Detailed metrics comparison")
        print("=" * 80)
        
        # Select main metrics for display (including new approximation error and full Hessian MSE)
        main_metrics = [
            'Mean_Frobenius_Error', 'avg_relative_frobenius_error', 
            'avg_relative_spectral_error', 'avg_directional_curvature_error',
            'avg_operator_norm_error', 'avg_diagonal_mse', 
            'avg_full_hessian_mse', 'avg_relative_full_hessian_mse'
        ]
        
        # Filter existing columns
        available_metrics = [col for col in main_metrics if col in df.columns]
        
        if available_metrics:
            print("\nMain metrics comparison:")
            print("-" * 80)
            display_df = df[['Method'] + available_metrics].copy()
            
            # Format numerical display
            for col in available_metrics:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            
            print(display_df.to_string(index=False))
        
        # Display full metrics per method
        print("\nFull metrics per method:")
        print("-" * 80)
        for _, row in df.iterrows():
            method = row['Method']
            print(f"\nMethod: {method}")
            print("-" * 40)
            
            # Display all available metrics
            for col in df.columns:
                if col not in ['Method'] and not col.startswith('Std_') and not col.startswith('Min_') and not col.startswith('Max_'):
                    value = row[col]
                    if not pd.isna(value):
                        print(f"  {col}: {value:.6f}")
                    else:
                        print(f"  {col}: N/A")
    
    def visualize_results(self, save_path: Optional[str] = None):
        """Visualize results (disabled)"""
        print("\nVisualization is disabled; only tabular results are shown")
        print("For visualization, please use other tools to process the CSV result file")
    
    def save_results(self, filename: str = 'hessian_simulation_results.csv'):
        """Save results to CSV file"""
        df = self.analyze_results()
        df.to_csv(filename, index=False)
        print(f"Results saved to: {filename}")
        return df

def main():
    """Main function - Run simulation according to user-specified test procedure"""
    print("=" * 80)
    print("Hessian Approximation Simulation")
    print("=" * 80)
    
    # Set parameters
    function_types = ['quadratic', 'rosenbrock',  'styblinski']
    dimensions = [5000]
    n_samples = 25
    random_seeds = [42, 123, 234, 345, 456, 567, 678, 789, 890, 901, 1012, 1214, 1416, 1618, 1820, 2022, 2224, 2426, 2628, 2830]  # Use multiple random seeds
    
    # Run simulation
    detailed_records = []
    
    # Run tests for each function type
    for func_type in function_types:
        print(f"\n{'='*80}")
        print(f"Test function: {func_type}")
        print(f"{'='*80}")
        
        for d in dimensions:
            print(f"\nDimension: {d}")
            print("-" * 40)
            
            # Run experiments for each seed
            for seed in random_seeds:
                print(f"\nUsing random seed: {seed}")
                
                # Initialize simulator
                simulator = HessianApproximationSimulator(function_type=func_type, d=d, n_samples=n_samples, seed=seed)
                
                # Sample points
                np.random.seed(seed)
                rng = np.random.default_rng(seed)  # Use seed for reproducibility
                z = rng.standard_normal(d)         # Equivalent to np.random.randn(d), but use rng for consistency
                u = z / np.linalg.norm(z)          # Uniform direction (on sphere)
                R = 100
                r = R * (rng.random() ** (1.0 / d))  # Radius: ensure "uniform inside sphere", not biased towards outer/inner layers
                center_point = u * r
        
                test_samples = simulator.sample_points(center_point)
                
                # Run methods
                methods = [
                    'CD',                         
                    'S1',                       
                    'S2',                
                    'S3',           
                    'ZoVH', 
                    'ZoVH_reuse' 
                ]
                method_results = {}
                
                for method in methods:
                    try:
                        method_result = run_single_method_comprehensive(
                            method, simulator.test_function, test_samples, K=3, d=d, noise_std=0.001, test_randomseeds=list(range(n_samples))
                        )
                        method_results[method] = method_result
                        print(f"Method {method} completed")
                    except Exception as e:
                        print(f"Method {method} failed: {str(e)}")
                        method_results[method] = {'error': str(e)}
                
                # Record error info for each method at each sample point
                for method in methods:
                    method_result = method_results.get(method)
                    if not method_result or 'error' in method_result:
                        continue

                    errors = method_result.get('errors', [])
                    randomseeds_list = method_result.get('sample_randomseeds', [])

                    for idx, (error_value, sample_randomseed) in enumerate(zip(errors, randomseeds_list)):
                        record = {
                            'Function': func_type,
                            'Dimension': d,
                            'Seed': seed,
                            'Method': method,
                            'Sample_Index': idx,
                            'Sample_RandomSeed': sample_randomseed,
                            'Frobenius_Error': error_value
                        }

                        detailed_records.append(record)

                print(f"Recorded error info for seed {seed}, total {len(detailed_records)} accumulated records")
    
    if detailed_records:
        details_df = pd.DataFrame(detailed_records)
        output_filename = 'hessian_simulation_detailed_errors.csv'
        details_df.to_csv(output_filename, index=False)
        print(f"\nDetailed error records saved to: {output_filename}")
    else:
        print("\nNo error data recorded")

    print(f"\n{'='*80}")
    print("Simulation completed!")
    print(f"{'='*80}")

def run_single_method_comprehensive(method, test_function, test_samples, K, d, noise_std=0.0, test_randomseeds=[]):
    """
    Run a single method's comprehensive test
    
    Args:
        method: Method name
        test_function: Test function
        test_samples: Test sample points
        test_values: Test sample points' function values
        d: Dimension
        noise_std: Noise standard deviation
        
    Returns:
        Result dictionary
    """
    results = {
        'method': method,
        'approximations': [],
        'errors': [],
        'sample_randomseeds': []
    }
    
    # Set method parameters
    mu = 0.1  # Perturbation parameter

    # Initialize previous Hessian (for BFGS and other iterative methods)
    previous_hessian = np.eye(d)  # Initialize with identity matrix
    
    for i, (sample_point, sample_randomseed) in enumerate(zip(test_samples, test_randomseeds)):
        # Get true Hessian matrix
        true_hessian = test_function.hessian_f(sample_point)
        
        # Choose approximation function based on method type
        if method == 'CD':
            approx_hessian = hessian_CD_approx(
                sample_point, test_function, d, K, mu, noise_std, sample_randomseed
            )
        elif method == 'S1':
            approx_hessian = hessian_S1_approx(
                sample_point, test_function, d, K, mu, noise_std, sample_randomseed
            )
        elif method == 'S2':
            approx_hessian = hessian_S2_approx(
                sample_point, test_function, d, K, mu, noise_std, sample_randomseed
            )
        elif method == 'S3':
            approx_hessian = hessian_S3_approx(
                sample_point, test_function, d, K, mu, noise_std, sample_randomseed
            )
        elif method == 'ZoVH':
            approx_hessian = hessian_ZoVH_approx(
                sample_point, test_function, d, K, mu, noise_std, sample_randomseed
            )
        elif method == 'ZoVH_reuse':
            history_values = prepare_history_values(test_samples, test_randomseeds, i)
            approx_hessian = hessian_ZoVH_reuse_approx(
                sample_point, test_function, d, K, mu, noise_std, history_values, sample_randomseed
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute evaluation metrics
        true_hessian = test_function.hessian_f(sample_point)

        frobenius_error = np.linalg.norm(true_hessian - approx_hessian, ord='fro')

        results['approximations'].append(approx_hessian)
        results['errors'].append(frobenius_error)
        results['sample_randomseeds'].append(sample_randomseed)
    
    return results

def prepare_history_values(samples, randomseeds,  current_index: int) -> Dict:
        """
        Prepare historical information for reuse methods
        
        Args:
            current_point: Current parameter point
            current_index: Current point's index
            
        Returns:
            history_values: Dictionary containing historical information
        """
        # Set number of historical iterations (including current iteration)
        N = 5 # Use up to N historical iterations
        
        # Prepare historical parameter vectors
        historical_thetas = []
        historical_randomseeds = []
        # Go back N-1 iterations from current point
        if current_index > 0:
            for n in range(1, N):
                if current_index - n >= 0:
                    # Use previous sample points as historical parameters
                    historical_thetas.append(samples[current_index - n].copy())
                    historical_randomseeds.append(randomseeds[current_index - n])
            
        
        return {
            'N': 1 + len(historical_thetas),
            'thetas': historical_thetas,
            'historical_randomseeds': historical_randomseeds
        }

def analyze_method_results(method_results, methods):
    """Analyze results and generate summary table"""
    print("\n" + "=" * 80)
    print("Results Analysis")
    print("=" * 80)
    
    # Create result summary table
    summary_data = []
    
    for method in methods:
        if method not in method_results or 'error' in method_results[method]:
            print(f"Method {method} has no valid error data, skipping statistics")
            continue
        
        results = method_results[method]
        errors = results['errors']
        metrics_list = results['metrics']
        
        # Check if there is valid error data
        if len(errors) == 0:
            print(f"Method {method} has no valid error data, skipping statistics")
            continue
        
        # Compute statistics
        mean_error = np.mean(errors)
        
        selected_metric_names = [
            'mean_frobenius_error',
        ]
        
        avg_metrics = {}
        for metric_name in selected_metric_names:
            values = []
            for metrics in metrics_list:
                if metric_name in metrics and not np.isnan(metrics[metric_name]):
                    values.append(metrics[metric_name])
            
            if values:
                avg_metrics[f'avg_{metric_name}'] = np.mean(values)
            else:
                avg_metrics[f'avg_{metric_name}'] = float('nan')
        
        summary_data.append({
            'Method': method,
            'Mean_Frobenius_Error': mean_error,
            **avg_metrics
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)  # Ensure return type is DataFrame
    
    # Ensure df is a DataFrame
    if isinstance(df, pd.DataFrame):
        # Sort by mean Frobenius error
        df = df.sort_values('Mean_Frobenius_Error')
    else:
        print("Warning: Returned result is not a DataFrame")
    
    print("\nMethod ranking (by mean Frobenius error):")
    print("-" * 80)
    print(df[['Method', 'Mean_Frobenius_Error']].to_string(index=False, float_format='%.4f'))
    
    return df  # Return DataFrame instead of dictionary


if __name__ == "__main__":
    main()
