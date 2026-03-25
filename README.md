# ML Concept Lab — BT3017 Project

An interactive web app for learning Kernel Trick, PCA, Spectral Clustering, and Graph Neural Networks.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open in browser
http://localhost:5050
```

## Structure

```
ml-concept-lab/
├── app.py              # Flask backend (all ML computation)
├── requirements.txt
└── templates/
    └── index.html      # Full frontend (single file)
```

## Features

Each of the 4 labs has:
- **① Explain** — 4-step animated explainer with plots
- **② Sandbox** — live interactive ML with real sklearn/numpy backend
- **③ Quiz** — 4-question quiz with explanations

## Labs

| Lab | Key interactions |
|-----|-----------------|
| Kernel Trick | Draw points, pick kernel, see decision boundary + 3D lift |
| PCA | Choose dataset, adjust components, see reconstruction error |
| Spectral Clustering | Compare spectral vs k-means side-by-side |
| GNN | Build graphs, run forward pass, animate message passing |
