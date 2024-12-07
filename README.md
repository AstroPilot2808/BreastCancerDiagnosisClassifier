Breast Cancer Diagnosis Classifier
Overview
This project focuses on building and evaluating different machine learning models for predicting breast cancer diagnosis based on a set of features. The goal is to identify which model provides the most accurate predictions and insights into the data.

Contents
Project Description
Installation
Data Preprocessing
Models
Naive Bayes
K-Nearest Neighbors (KNN)
Logistic Regression
Evaluation
Lessons Learned
Limitations
Conclusion
Project Description
This project involves building and evaluating three different machine learning models—Naive Bayes, K-Nearest Neighbors (KNN), and Logistic Regression—using the Breast Cancer Wisconsin (Diagnostic) Data Set from the UCI Machine Learning Repository. The dataset contains features such as radius, texture, perimeter, and area measurements, which are used to predict whether a breast tumor is malignant (M) or benign (B).

Objective
The primary goal is to determine which model provides the best prediction accuracy and AUC (Area Under the ROC Curve) score, while also offering interpretability and suitability for the given dataset.

Installation
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/breast-cancer-diagnosis-classifier.git
cd breast-cancer-diagnosis-classifier
Requirements:

Python (>=3.6)
Required Python libraries:
pandas
scikit-learn
matplotlib
You can install these requirements using:

bash
Copy code
pip install -r requirements.txt
Data Preprocessing
The dataset was preprocessed to:

Handle skewed features using log transformations.
Normalize the features to bring them to a similar scale.
Split the data into training and testing sets to evaluate the models.
Preprocessing Steps
Loading the dataset:

The dataset is loaded using pandas from a CSV file.
Initial data inspection and feature transformation are performed to address any skewness in the data using log transformations.
Feature Scaling:

Features were standardized using StandardScaler from scikit-learn to ensure that all features contribute equally to the model.
Splitting the Data:

The dataset is split into training (80%) and testing (20%) sets to evaluate model performance.
Models
Naive Bayes
Model: Gaussian Naive Bayes.
Performance:
Accuracy: 94.74%
AUC Score: 0.9918
Strengths:
Simple and interpretable model.
Robust performance on high-dimensional data.
Weaknesses:
Assumes independence between features which may not always hold true.
Less effective with non-linear relationships.
K-Nearest Neighbors (KNN)
Model: K-Nearest Neighbors with Euclidean distance.
Performance:
Accuracy: 92.11%
AUC Score: 0.9831
Strengths:
Can capture complex, non-linear relationships.
Simple to understand and implement.
Weaknesses:
Computationally expensive, especially with large datasets.
Sensitive to noise and outliers.
Logistic Regression
Model: Logistic Regression using the 'liblinear' solver.
Performance:
Accuracy: 92.98%
AUC Score: 0.9905
Strengths:
High interpretability.
Efficient and computationally fast.
Weaknesses:
Assumes a linear relationship between features and the log-odds.
May not handle complex interactions well.
Evaluation
Each model's performance was evaluated based on accuracy and AUC score. Naive Bayes showed the best performance overall with the highest accuracy and AUC score, making it the most suitable model for this dataset.

ROC Curves
ROC curves were plotted for each model to visualize their performance in distinguishing between malignant and benign cases.
Naive Bayes consistently had the best performance across all thresholds.
Lessons Learned
Model Selection: The choice of model is crucial depending on the dataset’s characteristics. Naive Bayes was most suitable due to the dataset's high-dimensional and complex nature.
Data Transformation: Transforming skewed features using log transformations significantly improved model performance.
Feature Engineering: Careful feature scaling and transformation were critical for optimal model performance.
Limitations
Dataset Size: The dataset is relatively small, which might not fully challenge the models.
Model Complexity: More sophisticated models like ensemble methods were not explored, which could have improved performance.
Assumptions in Models: Naive Bayes’ assumption of feature independence may not always be valid, potentially limiting its performance.
Conclusion
Naive Bayes emerged as the best-performing model for this breast cancer diagnosis task due to its robustness and efficiency in handling high-dimensional, complex data. Despite its limitations, its simplicity and strong performance make it a valuable tool for diagnostic purposes. Future improvements could include exploring more complex models and hyperparameter tuning to further refine predictions.

