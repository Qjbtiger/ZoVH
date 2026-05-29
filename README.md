# Revisiting Zeroth-Order Hessian Approximation: A Single-Step Policy Optimization Lens

This is official implementation for the paper "Revisiting Zeroth-Order Hessian Approximation: A Single-Step Policy Optimization Lens"

## Project Overview

This project proposes a unified view of Zeroth-Order Hessian approximation by recasting it as the Hessian of a smoothed, single-step policy-optimization objective. Building on this perspective, we introduce ZoVH, a variance-reduced suite of estimators for the full Hessian, its regularized inverse, and a bias-corrected inverse Hessian–gradient product. ZoVH combines an optimal averaged baseline with query reuse to improve sample efficiency without increasing query cost, delivering higher accuracy and faster convergence in practical tasks.


## Environment Setup

We recommend using `uv` to manage the Python environment:

```bash
uv sync && source ./.venv/bin/activate
```

## Experiments

### Empirical Hessian Error Analysis (Synthetic Funciton)

```bash
cd hessian_approx_error
python hessian_approximation_simulation.py
```

### Empirical Hessian Error Analysis (Neural Network)

First, trian the CNN model on MNIST dataset:

```bash
cd synthetic_and_adversarial
python train_mnist_hessian.py --config config/cnn_mnist.yaml
```

Then, analyze the results:

```bash
python analyze_cnn_results.py --input_dir "NEW result cnn" \
  --output "NEW result cnn/summary_by_layer_and_method.csv"
```


### Synthetic Functions Optimization

```bash
cd synthetic_and_adversarial
python run_synthetic.py --config config/synthetic.yaml
```

### Adversarial Attack on MNIST

```bash
cd synthetic_and_adversarial
python run_adversarial.py --config config/adversarial.yaml
```

### Memory-Efficient LLM Fine-Tuning

```bash
cd mezo
bash zovh.sh
```