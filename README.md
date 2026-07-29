# 🔥 Structure Fire Performance Predictor

This repository contains the machine learning models and an interactive inference pipeline for predicting the fire performance and damage state of composite flooring systems. 

Utilizing LightGBM (LGBM) classification and regression models, this tool allows researchers and engineers to input specific structural and thermal parameters to instantly predict slab displacement and overall damage states under fire conditions.

## 📊 Features

The repository includes a Jupyter Notebook with a fully interactive UI built using `ipywidgets`. It dynamically extracts feature distributions from the dataset and provides bounded input fields to prevent erroneous data entry. 

**Model Inputs (Features):**
*   `A_s`
*   `AP`
*   `reX` & `reY`
*   `Fire load`
*   `Opening factor`
*   `Conductivity`
*   `Specific heat`
*   `Density`
*   `Yield stress`

**Model Outputs (Targets):**
1.  **Damage State:** Categorical prediction of the structural damage state (LGBM Classifier).
2.  **Max Displacement at Slab Center:** Continuous prediction of the maximum displacement (LGBM Regressor).
3.  **Residual Displacement:** Continuous prediction of the slab center displacement at the final time step (LGBM Regressor).

## 📁 Repository Structure

*   `Class_model_LGBM_Classifier_Damage_state.pkl`: Trained classification model for predicting the damage state.
*   `Reg_model_LGBM_Regressor_max_displacement_at_slab_center.pkl`: Trained regression model for maximum displacement.
*   `Reg_model_LGBM_Regressor_slab_center_displacement_at_the_last_time_step_residual.pkl`: Trained regression model for residual displacement.
*   `dataset_test.csv`: Testing dataset used as a template to define the interactive UI's feature bounds and dropdown options.
*   `inference_notebook.ipynb`: The main Jupyter Notebook containing the prediction logic and interactive GUI.

## 🚀 Usage & Installation

**Prerequisites:**
Ensure you have Python 3.8+ installed along with the following packages:
```bash
pip install pandas numpy scikit-learn lightgbm joblib ipywidgets