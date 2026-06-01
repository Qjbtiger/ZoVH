<div align="center">

<h1>ZoVH</h1>

<h3>Revisiting Zeroth-Order Hessian Approximation: A Single-Step Policy Optimization Lens</h3>


<p>
  <a href="https://icml.cc/virtual/2026/poster/61753"><img src="https://img.shields.io/badge/Paper-ICML%202026-blue" alt="ICML 2026"></a>
</p>

</div>

We proposes a unified view of Zeroth-Order Hessian approximation by recasting it as the Hessian of a smoothed, single-step policy-optimization objective. Building on this perspective, we introduce ZoVH, a variance-reduced suite of estimators for the full Hessian, its regularized inverse, and a bias-corrected inverse Hessian-gradient product. ZoVH combines an optimal averaged baseline with query reuse to improve sample efficiency without increasing query cost, delivering higher accuracy and faster convergence in practical tasks.

## Setup

```bash
uv sync
source ./.venv/bin/activate
```

## Experiments

### Hessian Error on Synthetic Functions

```bash
cd hessian_approx_error
python hessian_approximation_simulation.py
```

### Hessian Error on Neural Networks

```bash
cd synthetic_and_adversarial
# First, train the CNN model on MNIST dataset:
python train_mnist_hessian.py --config config/cnn_mnist.yaml
# Then, analyze the results:
python analyze_cnn_results.py --input_dir "NEW result cnn" \
  --output "NEW result cnn/summary_by_layer_and_method.csv"
```

### Synthetic Function Optimization

```bash
cd synthetic_and_adversarial
python run_synthetic.py --config config/synthetic.yaml
```

### MNIST Adversarial Attack

```bash
cd synthetic_and_adversarial
python run_adversarial.py --config config/adversarial.yaml
```

### Memory-Efficient LLM Fine-Tuning

```bash
cd mezo
bash zovh.sh
```

## Citation

```bibtex
@inproceedings{zovh,
  title = {Revisiting Zeroth-Order Hessian Approximation: A Single-Step Policy Optimization Lens},
  author = {Qiu, Junbin and Hong, Zhaowei and Xu, Renzhe and Shu, Yao},
  booktitle = {Forty-third International Conference on Machine Learning},
  year = {2026}
}
```
