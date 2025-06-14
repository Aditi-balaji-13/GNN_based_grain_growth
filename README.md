# Graph Neural Network-Based Grain Growth Simulations

This repository contains select code and analysis from my final B.Tech Project, which focuses on modeling grain growth simulations using Graph Neural Networks (GNNs). The goal of the project is to analyze and predict the evolution of grain structures over time using graph-based representations derived from image tracking data.

---

## 📁 Project Structure

```text
├── Data Handling
│   ├── 1_Data_adj_and_triple_point.ipynb
│   ├── 2_Data_Visualisation.ipynb
│   ├── 3_tracked_points_evaluation.ipynb
│   ├── 4_gnn_model.ipynb
│   ├── 8_tracked_value_processing.ipynb
│   └── 9_failed_Image_tracking.ipynb
├── README.md
├── __pycache__
│   ├── model.cpython-38.pyc
│   └── utils.cpython-38.pyc
├── adj_mat.pickle
├── main.py
├── model.py
├── test_main.py
├── tracks.pickle
└── utils.py
```

---

## 🧠 Project Overview

Grain growth in materials is a critical phenomenon influencing the mechanical properties of metals. This project uses GNNs to learn and predict changes in grain structures over time by converting microstructural image data into graph representations. The nodes represent grains or junctions, and the edges represent grain boundaries.

---

## 🧪 Key Components

- **Data Processing**: Extracts adjacency matrices and relevant features from image-tracked grain data.
- **Visualization**: Uses plotting utilities to visually verify graph structures and evolution.
- **Modeling**: Implements a custom GNN architecture to predict grain boundary dynamics.
- **Testing**: Includes test scripts for debugging and ensuring reproducibility.

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- NumPy, Matplotlib
- NetworkX
- Pickle

To install dependencies:

```bash
pip install -r requirements.txt
```
---

## 🚀 Usage
```bash
# To run the model
python main.py

# To run tests
python test_main.py
```
---
## 👩‍💻 Author
Aditi Balaji \
B.Tech in Metallurgical and Materials Engineering \
Indian Institute of Technology, Madras
---
## 📄 License
This code is for academic and non-commercial research purposes only.
 
