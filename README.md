# 🧠 Neural Network from Scratch in C

**Training on California Housing Dataset (`housing.csv`)**

---

## 📌 Project Overview

This project implements a fully connected feedforward neural network **from scratch in C** without using any external machine learning libraries.

The model is trained on the **California Housing dataset** to predict median house values using structured tabular data.

The purpose of this project is to:

* Understand neural networks at a low level
* Implement forward and backward propagation manually
* Build a real training engine in pure C
* Train on a real-world dataset (~20,000 samples)
* Apply core ML engineering practices (normalization, shuffling, validation, regularization)

This is not a toy implementation — it trains on real data and supports proper evaluation.

---

## 📂 Dataset

**File:** `housing.csv`

Features (8 input variables):

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income

Target (1 output variable):

* median_house_value

The model performs **regression** using Mean Squared Error (MSE) loss.

---

## 🏗 Architecture

The neural network architecture:

```
Input Layer (8 features)
        ↓
Hidden Layer (16 neurons, ReLU activation)
        ↓
Output Layer (1 neuron, Linear activation)
```

### Configuration:

* `INPUT_SIZE = 8`
* `HIDDEN_SIZE = 16`
* `OUTPUT_SIZE = 1`

---

## ⚙️ Features Implemented

### 1️⃣ CSV Dataset Loading

* Reads `housing.csv`
* Skips header row
* Parses input features and target
* Supports up to 25,000 samples

---

### 2️⃣ Data Preprocessing

* Feature normalization (Min–Max scaling)
* Train/Validation split (80% / 20%)
* Dataset shuffling at every epoch

Normalization is computed using training data statistics to prevent data leakage.

---

### 3️⃣ Weight Initialization

He initialization is used for stability with ReLU:

```
scale = sqrt(2 / fan_in)
```

This improves convergence and prevents exploding/vanishing gradients.

---

### 4️⃣ Forward Propagation

Hidden layer:

```
z = W·x + b
a = ReLU(z)
```

Output layer:

```
y = W·a + b
```

---

### 5️⃣ Backpropagation

Manual gradient computation:

* Output error calculation
* Hidden layer error propagation
* Weight and bias updates
* L2 regularization

Loss function:

```
MSE = (y_pred - y_true)^2
```

---

### 6️⃣ Regularization

L2 regularization is applied during weight updates:

```
weight -= learning_rate * (gradient + λ * weight)
```

This helps reduce overfitting on large datasets.

---

### 7️⃣ Validation Monitoring

After each epoch:

* Training Loss is calculated
* Validation Loss is calculated
* Both are printed

This allows detection of overfitting.

---

## 🧪 Training Process

For each epoch:

1. Shuffle training data
2. Perform forward pass
3. Compute loss
4. Backpropagate errors
5. Update weights
6. Evaluate validation loss

---

## 📊 Example Output

```
Epoch 0 | Loss 53064987131.776009
Epoch 1 | Loss 47549253482.912666
Epoch 2 | Loss 28196656985.694962
Epoch 3 | Loss 14598957015.256060
Epoch 4 | Loss 13277542323.550255
Epoch 5 | Loss 13212601290.228926
```

---

## 🛠 How to Compile

```bash
gcc NeuralNetwork.c -o nn -lm
```

---

## ▶ How to Run

Make sure `housing.csv` is in the same directory:

```bash
./nn
```

---

## 📁 Project Structure

```
.
├── neural_network.c
├── housing.csv
└── README.md
```

---

## 🎯 What This Project Demonstrates

This project shows practical understanding of:

* Matrix operations in C
* Manual gradient descent
* ReLU activation
* L2 regularization
* Data normalization
* Train/validation splitting
* Real dataset training
* Memory-efficient static arrays
* Numerical stability in ML

---

## 🚀 Next Improvements (Planned)

The following upgrades are planned:

* Fully dynamic layer engine (arbitrary depth)
* Adam optimizer
* Multi-layer deep network support
* Model saving/loading
* Performance optimization (OpenMP / optimized matrix ops)
* Dynamic memory allocation (malloc-based architecture)
* Batch training

---

## 👨‍💻 Why This Matters

Most ML work today happens inside high-level libraries like:

* PyTorch
* TensorFlow
* Scikit-learn

For this project I wanted it to work out mainly on low-level program.

Understanding this level gives:

* Deep intuition about gradient flow
* Better debugging ability
* Strong systems-level ML engineering skills
* Performance awareness

---

## 📌 Author

Clifford Odiwuor Ojuka
Machine Learning & Systems Engineering Enthusiast

---
