# cardiovascular-disease

The goal of this project is to analyze and predict cardiovascular diseases based on patient's health conditions. It uses the [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) from Kaggle. 

**datapreparation.py**: 
- Cleans the data and filters out false values
- Builds heatmaps, histograms, and plots to understand the correlation and distribution of the features

**model.py**: 
- Uses a random forest classifier to predict cardiovascular diseases
- Finds the optimal values for the algorithm's hyperparameters
