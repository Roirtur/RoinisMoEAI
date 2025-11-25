# Roadmap: Mixture of Experts Project

## 1. Model Implementation

### `models/`
- [ ] Implement the expert models in separate files inside the `models` directory.
- [ ] Use `models/__init__.py` to easily import the experts.

### `moe_model.py`
- [ ] Implement the gating network `g(x)` that produces a distribution over the experts.
- [ ] Implement the main MoE model that combines the experts from the `models` directory and the gating network.
- [ ] Implement the combination of outputs (soft routing or hard routing).
- [ ] Choose the number of experts, layer sizes, and routing strategy.

## 2. Data and Task
- [ ] Choose a suitable task and dataset:
    - **Vision:** MNIST / FashionMNIST / CIFAR-10 (Classification)
    - **NLP (optional):** IMDB / AG News (Sentiment Analysis / Categorization)
    - **Tabular:** UCI Games (Wine, Iris, etc.) (Regression / Classification)
- [ ] Implement data loading and preprocessing pipelines.

## 3. Training (`train.py`)
- [ ] Implement the main training script.
- [ ] Train the MoE model.
- [ ] For comparison, train a dense network of equivalent capacity.

## 4. Experiments and Analysis (`experiments.ipynb`)
- [ ] Compare your MoE with the equivalent dense network.
- [ ] Study the impact of:
    - [ ] The number of experts.
    - [ ] The type of gating (soft vs. hard).
    - [ ] Regularization (dropout, gating entropy, etc.).
- [ ] Visualize:
    - [ ] The activation distribution of the experts.
    - [ ] The evolution of the losses of each expert.
    - [ ] The distribution of data processed by each expert.

## 5. Report (`report.tex`)
- [ ] Write an introduction and a brief review of the MoE concept.
- [ ] Describe your model (architecture, design choices).
- [ ] Detail the experimental protocol and the hyperparameters used.
- [ ] Present the experimental results and their interpretation.
- [ ] Write a critical discussion (strengths, limitations, perspectives).
- [ ] Compile the `.tex` file to generate `report.pdf`.

## 6. Optional Extensions
- [ ] Top-k routing with Gumbel-Softmax or Straight-Through Estimator.
- [ ] Sparsity regularization on gating.
- [ ] Visualization of routing by expert (expert ↔ classes heatmap).
- [ ] Insertion of an MoE layer in a small Transformer.
