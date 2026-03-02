# 🧠 Mini Deep Learning Training Engine in C

**Training on the California Housing Dataset (`housing.csv`)**

---

# 📌 Project Overview

This project implements a **mini deep learning training engine written entirely in C** without relying on external machine learning libraries.

The goal of this project is to **understand and implement neural networks at the systems level**, including forward propagation, backpropagation, optimization, and dataset management.

Unlike many toy examples, this engine trains on a **real-world dataset (California Housing)** containing thousands of samples.

The project demonstrates how modern deep learning systems work **under the hood**, including:

* neural network architecture construction
* gradient-based optimization
* mini-batch training
* data preprocessing and normalization
* model persistence

This implementation focuses on **clarity, correctness, and performance**, making it useful both as a learning resource and a systems-level ML engineering project.

---

# 📂 Dataset

**File:** `housing.csv`

The dataset contains **California housing statistics** used to predict median house value.

### Input Features (8)

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income

### Target

* median_house_value

The model performs **regression** using **Mean Squared Error (MSE)** loss.

Dataset size: **~20,000 samples**

---

# 🏗 Neural Network Architecture

The engine supports **fully configurable multi-layer networks**.

Example architecture used in this project:

```
Input Layer (8 features)
        ↓
Hidden Layer (64 neurons, ReLU)
        ↓
Hidden Layer (32 neurons, ReLU)
        ↓
Output Layer (1 neuron, Linear)
```

Layers can be easily modified inside the architecture configuration.

---

# ⚙️ Core Features Implemented

## 1️⃣ Dynamic Layer Engine

The network architecture is defined dynamically using a layer configuration array:

```
int layer_sizes[] = {8, 64, 32, 1};
```

This allows the model to support **arbitrary depth neural networks** without rewriting training logic.

---

# 2️⃣ Adam Optimizer

The training engine implements the **Adam optimization algorithm**, one of the most widely used optimizers in deep learning.

Adam combines:

* Momentum
* Adaptive learning rates

Update rule:

```
m = β1 * m + (1 - β1) * g
v = β2 * v + (1 - β2) * g²
w = w - lr * m̂ / (sqrt(v̂) + ε)
```

This significantly improves training stability and convergence speed.

---

# 3️⃣ Mini-Batch Training

Training is performed using **mini-batches**, which improves:

* gradient stability
* convergence speed
* numerical efficiency

Each epoch:

1. Dataset is shuffled
2. Data is processed in batches
3. Gradients are accumulated
4. Parameters are updated

---

# 4️⃣ Dataset Shuffling

Training samples are **randomly shuffled each epoch**.

Benefits:

* Prevents model bias from sample ordering
* Improves generalization
* Stabilizes training

---

# 5️⃣ Feature Normalization

Input features are normalized using **Min–Max scaling**:

```
x_norm = (x - min) / (max - min)
```

This ensures features operate within the same range, which improves gradient descent performance.

---

# 6️⃣ Model Saving & Loading

The engine supports **model persistence**.

Trained weights can be saved to disk:

```
model.bin
```

This allows:

* reloading trained models
* skipping retraining
* deployment experiments

---

# 7️⃣ Performance Optimizations

Several optimizations were implemented:

* efficient memory layout
* preallocated gradient buffers
* batch computation loops
* reduced memory allocation overhead

The implementation remains **fast while staying readable**.

---

# 🧪 Training Process

For each epoch:

1. Shuffle dataset
2. Create mini-batches
3. Perform forward propagation
4. Compute loss
5. Backpropagate gradients
6. Update parameters using Adam
7. Report training loss

---

# 📊 Example Training Output

```
Epoch 0 | Loss 53064987131.776009
Epoch 1 | Loss 47549253482.912666
Epoch 2 | Loss 28196656985.694962
Epoch 3 | Loss 14598957015.256060
Epoch 4 | Loss 13277542323.550255
Epoch 5 | Loss 13212601290.228926
```

Loss steadily decreases as the model learns the dataset patterns.

---

# 🛠 How to Compile

Compile using GCC:

```bash
gcc NeuralNetwork.c -o nn -lm -fopenmp
```

---

# ▶ How to Run

Place `housing.csv` in the project directory and run:

```bash
./nn
```

---

# 📁 Project Structure

```
.
├── NeuralNetwork.c
├── housing.csv
├── model.bin
└── README.md
```

---

# 🎯 What This Project Demonstrates

This project highlights practical knowledge of:

* neural networks from first principles
* gradient descent and backpropagation
* optimizer implementation (Adam)
* dataset handling in C
* numerical computation
* mini-batch training
* memory management in C
* systems-level machine learning engineering

---

# 🚀 Project Milestone

This project represents **Milestone 1** in a larger research direction:

**Building machine learning systems from scratch in low-level languages.**

Future milestones will extend this work into a **modular ML framework**.

---

# 🔬 Future Work

Planned extensions include:

* Modular ML framework architecture
* Automatic differentiation (Autograd)
* CUDA acceleration
* SIMD vectorization
* Softmax classification support
* Dropout and regularization layers
* Visualization tools for training metrics

---

# 👨‍💻 Author

**Clifford Odiwuor Ojuka**
Machine Learning & Systems Engineering Enthusiast

---

💡 *This project was built to explore how modern deep learning systems operate internally rather than relying solely on high-level frameworks.*