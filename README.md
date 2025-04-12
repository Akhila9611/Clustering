# 🧠 Customer Clustering App


---

## 📌 Overview

This project implements a **customer segmentation model** using unsupervised learning (clustering). It analyzes customer characteristics from mall data to group similar individuals, helping businesses understand their customer base.

An interactive Streamlit app is available for testing the clustering model with user input or batch data.

---

## 🎯 Goal

- Segment customers using **clustering techniques** (e.g., KMeans)
- Enable real-time predictions using an app interface
- Visualize feature contributions and cluster behavior

---

## 🤖 Machine Learning Summary

- **Task Type:** Unsupervised Learning (Clustering)  
- **Target:** Identify customer segments  
- **Key Evaluation Metrics:** Inertia, Silhouette Score (used during development)

---

## 🗂️ Project Structure

```
CLUSTERING/
├── data/
│   ├── raw/
│   │   └── mall_customers.csv         # Raw input dataset
│   └── processed/
│       └── preprocessed.csv           # Cleaned, transformed dataset
│
├── src/
│   ├── data/
│   │   └── make_dataset.py            # Script to load and preprocess data
│   ├── features/
│   │   └── build_features.py          # Feature engineering and scaling
│   ├── models/
│   │   ├── train_model.py             # Train and save clustering model
│   │   ├── predict_model.py           # Load model and predict clusters
│   │   ├── kmeans_model.pkl           # Saved trained KMeans model
│   │   └── scaler.pkl                 # Saved StandardScaler object
│   └── visualization/
│       └── (Add plotting scripts here)
│
├── app.py                             # Streamlit app for user interaction
├── main.py                            # Main pipeline runner script
├── requirement.txt                    # Project dependencies
└── README.md                          # This documentation
```

---