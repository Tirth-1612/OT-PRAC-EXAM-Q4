<div align="center">

# 🌾 Smart Crop Assignment
### Machine Learning + Quantum Optimization + Parallel Computing

*Which crop should go on which plot, with no same crop on neighbouring plots?*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVR-orange?logo=scikitlearn&logoColor=white)
![Quantum](https://img.shields.io/badge/Quantum-QAOA%20p%3D1-blueviolet)
![Parallel](https://img.shields.io/badge/Parallel-ProcessPoolExecutor-green)
![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)

</div>

---

## 📖 Problem

A farm has **4 plots** in a row and **3 crops** (Wheat, Rice, Maize). Each plot gets exactly one crop, and **neighbouring plots must not grow the same crop** (rotation rule).

The project combines three techniques:

| Stage | Technique | Role |
|:--:|---|---|
| **a** | 🤖 Support Vector Regression | Predicts the expected yield of every crop on every plot |
| **b** | 🧮 QUBO | Turns the decision into an energy function with 12 binary variables |
| **c** | ⚛️ QAOA (p = 1) | Finds the lowest-energy assignment on a simulator |
| **d** | ⚡ Parallel computing | Evaluates all 2¹² bit-strings across 1, 2 and 4 workers |

---

## 🗂️ Repository Structure

```
├── farm_project.py      # full script (parts a–e)
├── farm_project.ipynb   # same code as a notebook (Colab-ready)
├── requirements.txt     # numpy, scikit-learn, scipy
└── README.md
```

---

## 🚀 Quick Start

**Option 1: Google Colab**
1. Open [colab.research.google.com](https://colab.research.google.com).
2. Choose **File → Open notebook → GitHub** and paste this repo's URL.
3. Choose **Runtime → Run all**.

**Option 2: Local**
```bash
git clone https://github.com/YOUR_USERNAME/farm-crop-qaoa.git
cd farm-crop-qaoa
pip install -r requirements.txt
python farm_project.py
```

> QAOA is simulated in plain NumPy, so **no Qiskit install is needed**.

---

## 🔬 Method

### a) Machine Learning
- 600 synthetic rows, `random seed = 23`, **70/30 train-test split**.
- Features: soil nitrogen, rainfall, plot area, crop, previous crop.
- Model: `SVR(kernel="rbf", C=10, epsilon=0.05)` with scaling and one-hot encoding.
- Output: a 4 × 3 table of predicted yields.

### b) QUBO

$$
E(x) = -\sum_{p,c} Y_{pc}\,x_{pc} \;+\; A\sum_{p}\Big(\sum_{c}x_{pc}-1\Big)^2 \;+\; B\sum_{(p,q)\in N}\sum_{c}x_{pc}\,x_{qc}
$$

- `x[p,c] = 1` means plot *p* gets crop *c* (4 × 3 = **12 qubits**).
- Term 1: maximise total predicted yield.
- Term 2: **one crop per plot** (penalty A).
- Term 3: **no same crop on neighbours** (penalty B).
- A = B = 2 × the maximum predicted yield.
- The result is checked against brute force over all 4096 states and against a direct search of the valid plans.

### c) QAOA (p = 1)
- Starts from |+⟩¹², applies a cost layer `exp(-iγE)` and a mixer `RX(2β)`.
- Angles (γ, β) are found by grid search followed by Nelder-Mead.
- 8192 simulated shots, with the best bit-string, its energy and its frequency reported.

### d) Parallel Computing
- All 2¹² energies are split into equal chunks.
- Run with `concurrent.futures.ProcessPoolExecutor` using 1, 2 and 4 workers.
- Speed-up = T₁ / Tₙ, Efficiency = Speed-up / n.

---

## 📊 Results

> Numbers below come from one run. Timings change from machine to machine.

**Machine learning**

| Metric | Value |
|---|:--:|
| RMSE | 0.365 |
| R² | 0.938 |

**Best assignment (brute force = QAOA)**

| Plot 1 | Plot 2 | Plot 3 | Plot 4 |
|:--:|:--:|:--:|:--:|
| 🍚 Rice | 🌽 Maize | 🌾 Wheat | 🍚 Rice |

- Lowest energy: **−32.167** (total predicted yield 32.167)
- QAOA found this answer in **36 / 8192** shots (0.44%), about **19×** more often than a random guess.

**Parallel performance**

| Workers | Time (s) | Speed-up | Efficiency |
|:--:|:--:|:--:|:--:|
| 1 | 0.175 | 1.00 | 100% |
| 2 | 0.107 | 1.64 | 82% |
| 4 | 0.117 | 1.50 | 38% |

---

## 🧠 What I Learned

1. SVR gave reliable yield predictions (R² ≈ 0.94), which made the QUBO costs meaningful.
2. QAOA with p = 1 found the same optimum as brute force, but the best state is only a small fraction of the shots.
3. Parallel speed-up depends on the number of CPU cores and on how much work each chunk does. Process start-up cost lowers efficiency at 4 workers on 2 cores.

---

## 🛠️ Tech Stack

`Python` · `NumPy` · `scikit-learn` · `SciPy` · `concurrent.futures`

---

<div align="center">

Made for **Q.4: ML + Quantum + Parallel Computing**

</div>