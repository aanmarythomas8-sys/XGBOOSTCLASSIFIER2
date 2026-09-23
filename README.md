# XGBoost Classifier Lab 🚀

An interactive, browser-based Machine Learning experimentation platform designed to demonstrate the mechanics, training pipeline, and hyperparameter tuning of an **XGBoost Classifier**.

---

## 🌟 Key Features

- **In-Browser Machine Learning Execution Engine:** Pure client-side JavaScript implementation of Gradient Boosted Decision Trees (GBDT).
- **Interactive Hyperparameter Tuning:** Adjust `N Estimators`, `Learning Rate`, `Max Depth`, `Subsample`, `Colsample Bytree`, and `Regularization (Alpha)` in real-time.
- **VS Code / Developer Terminal Aesthetic:** Dark-mode interface complete with animated execution outputs.
- **Dynamic Data Explorer:** Sortable, searchable, and paginated view of the dataset.
- **Comprehensive Evaluation Visualizations:**
  - Dynamic Metrics Cards (Accuracy, Precision, Recall, F1 Score, ROC-AUC)
  - Interactive Confusion Matrix
  - Feature Importance Horizontal Bar Chart
  - Training Loss / LogLoss Trajectory Curve
  - Predicted Class Probability Scatter Distribution
- **Live Prediction Playground:** Input custom feature measurements to get real-time model predictions and confidence probabilities.
- **Python Code Generator:** Instant copy-pasteable Scikit-Learn/XGBoost Python script matching the configured experiment.

---

## 📊 Dataset Specification

This project utilizes key diagnostic features derived from the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset:

- **Target variable:** `diagnosis` (`0` = Benign, `1` = Malignant)
- **Features included:**
  1. `radius_mean`
  2. `texture_mean`
  3. `perimeter_mean`
  4. `area_mean`
  5. `smoothness_mean`
  6. `compactness_mean`
  7. `concavity_mean`
  8. `concave_points_mean`
  9. `symmetry_mean`
  10. `fractal_dimension_mean`

---

## 🔄 Machine Learning Execution Pipeline

```text
[ Raw CSV Data ] ➔ [ 80/20 Stratified Split ] ➔ [ Gradient Boosting Ensembling ] ➔ [ Evaluation Metrics ]
