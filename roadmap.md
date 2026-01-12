# Mixture of Experts (MoE) Project Roadmap

## 1. Project Architecture & Setup
The following directory structure is required to maintain modularity and allow parallel work.

```text
project_moe/
├── data/                   # Dataset storage
├── models/
│   ├── __init__.py
│   ├── dense_baseline.py   # Standard network for comparison
│   ├── experts.py          # Definition of Expert layers (MLP/CNN)
│   ├── gating.py           # Definition of Gating mechanism (Soft/Hard)
│   └── moe_model.py        # Assembling Experts + Gating
├── report/                 # Report source files
│   └── report.tex
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # Data preprocessing and loading
│   └── visualization.py    # Plotting tools for the report
├── experiments.ipynb       # Notebook for analysis and visual generation
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── roadmap.md              # Project roadmap
└── train.py                # Main training loop
```

## 2. Phase I: Core Implementation (Week 1)

**Objective:** Establish the codebase and get a basic MoE running.
**Strategy:** Split the model components (Person A) from the infrastructure (Person B) to avoid merge conflicts.

### Setup

* [x] Initialize Git repository.
* [x] Create the file structure listed above.
* [x] Select Dataset: **CIFAR-100**.

### Workstream A: Model Architecture

**Owner: Person A**

* [x] **Implement Experts (`models/experts.py`):**
* Defines `ExpertLayer` as a distinct "Small CNN" (e.g. `Conv -> ReLU -> Pool`).
* Each expert is capable of classifying the image into classes (100 classes).
* Input: Image Batch. Output: Class Logits.


* [x] **Implement Gating (`models/gating.py`):**
* Defines `GatingNetwork` as a "Tiny CNN" router.
* Rapidly downsamples the image and outputs Softmax probability weights.
* Goal: Low cost classification of image "type".


* [x] **Assemble MoE (`models/moe_model.py`):**
* Implements the "Manager" for Conditional Computation.
* **Hard Routing Logic:**
    1. Router selects Top-1 expert per image.
    2. Manager splits the batch.
    3. Only the selected Expert runs on its assigned images (Saving FLOPs).
    4. Outputs are recombined.
* Allow the number of experts () to be a parameter.





### Workstream B: Infrastructure & Baseline

**Owner: Person B**

* [x] **Data Pipeline (`utils/data_loader.py`):**
* Implement PyTorch/TensorFlow dataloaders for the chosen dataset.
* Ensure train/validation/test splits are correct.


* [x] **Dense Baseline (`models/dense_baseline.py`):**
* Implemented a **SimpleBaseline** style architecture (matching ExpertLayer structure).
* Created `SimpleBaseline` class with a `width_multiplier` argument.
* This allows creating both:
    *   *Iso-FLOPs Baseline:* Reduce width ($k < 1.0$) to match active experts.
    *   *Iso-Params Baseline:* Increase width ($k \ge 1.0$) to match total MoE storage.
* Includes a helper `count_parameters()` method for exact comparison.




* [x] **Training Loop (`train.py`):**
* Implemented separate `train_baseline` function with full training logic (SGD + Cosine Scheduling).
* Added `train_moe` placeholder for future implementation.
* Added `main` with `argparse` to select model type and save path.
* Supports saving/loading logic to avoid retraining.
* *Note:* Ensure you can mask loss based on gating decisions to track "Loss per Expert".
* 
*Extension:* Add logic to track "Expert Usage" (frequency of selection) during training.





## 3. Phase II: Experimentation & Variations (Week 2)

**Objective:** Generate data for the report. You need to compare methods and visualize internals.
**Strategy:** Person A refines the MoE logic; Person B runs the heavy training and visualization code.

### Workstream A: Advanced MoE Logic

**Owner: Person A**

* [x] **Regularization & Load Balancing:**
    * Add logic to penalize low entropy in gating (prevent collapse to a single expert).
    * Implement a "Load Balancing" loss to ensure experts are utilized evenly (e.g., Coefficient of Variation).

* [ ] **Top-k > 1 Variation:**
    *   Extend `moe_model.py` to support Top-k selection (e.g. Top-2) where two experts run per image.
    *   Goal: Analyze the trade-off between Accuracy gain vs Inference Cost increase.

### Workstream B: Execution & Visualization

**Owner: Person B**

* [ ] **Visualization Tools (`utils/visualization.py`):**
    * Create functions to plot:
        * Expert activation distribution (Histogram).
        * Loss curves per expert.
        * Heatmaps of (Class vs. Expert) to show specialization.

* [ ] **Run Experiments (comparative study):**
    * Run 1: Dense Baseline (ResNet-18).
    * Run 2: **MoE - Conditional Computation (Top-1)** [Main approach].
    * Run 3: MoE with Top-2 Routing (if implemented).
    * Run 4: MoE with varying experts (e.g., 4 vs 8).





## 4. Phase III: Analysis & Reporting (Week 3)

**Objective:** Synthesize findings into the deliverables.
**Strategy:** Split the writing based on the technical implementation work.

### Analysis Tasks (Joint)

* [ ] **Specialization Analysis:** Check if specific experts handle specific classes (e.g., Expert 1 handles "Shoes", Expert 2 handles "Shirts").


* [ ] **Performance vs. Cost:** specific analysis on inference cost vs accuracy.

Report Writing (4-6 Pages) 

* [ ] **Introduction & MoE Concept:** Explain Conditional Computation and Routing [Person A].
* [ ] **Model Description:** Diagram of the architecture and math formulation () [Person A].
* [ ] **Protocol:** Dataset details, hyperparameters, and hardware used [Person B].
* [ ] **Results & Interpretation:**
* Compare Baseline vs MoE accuracy.
* Discuss the impact of Gating (Soft vs Hard) [Person B].
* Show visualizations of Expert Loading [Person B].


* [ ] **Critical Discussion:** discuss limits (e.g., collapse, training instability) [Person A].

### Final Polish

* [ ] **Code Cleanup:** Add comments and docstrings.
* [ ] **Presentation:** Prepare 5-6 slides for the 10-minute oral defense.


* [ ] **Zip Deliverables:** Ensure `code + report + slides` are ready.

```
