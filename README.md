# **ML Model for Cancer Detection using Python**
## **Overview**
This project focuses on developing a machine learning model to classify cancer as "Benign" or "Malignant" based on human cell sample records. The data used for this project is sourced from the UCI Machine Learning Repository (Asuncion and Newman, 2007), containing several hundred human cell records. The goal is to build an accurate predictive model that can assist in early cancer detection.

## **Project Summary**
Given that this is a classification problem, a Support Vector Machine (SVM) was chosen as the primary algorithm due to its robustness in handling high-dimensional data. SVM offers various kernel functions, and in this project, four different kernels were evaluated:

* Linear Kernel
* Polynomial Kernel
* Radial Basis Function (RBF) Kernel
* Sigmoid Kernel
  
The models were trained and tested by splitting the dataset into training and testing subsets. The best model was selected based on performance metrics such as Accuracy, F1-Score, and Jaccard Index.

## **Key Results:**
* Best Model: SVM with Polynomial Kernel
* Accuracy: 96.35%
* Average F1-Score: 0.9631
* Jaccard Score: 0.9444
  
## **Process Breakdown**
1. Data Wrangling
* Data Retrieval and Preparation:
* Collected and preprocessed the dataset from the UCI Machine Learning Repository.
* Handled missing values, normalized data, and ensured that the dataset was clean and ready for analysis.
2. Exploratory Data Analysis (EDA)
* Correlation Analysis: Performed correlation analysis to identify the relationship between features and the target variable.
* Distribution Analysis: Visualized the distribution of features to understand their variability and impact on the classification task.
3. Model Analysis
* Data Splitting: Split the dataset into training and testing subsets to validate model performance.
* Model Development: Developed four SVM models, each with a different kernel function (Linear, Polynomial, RBF, Sigmoid).
* Grid Search: Used grid search to fine-tune the hyperparameters of each model, optimizing their performance.
4. Model Evaluation
* Confusion Matrix: Constructed confusion matrices for each model to assess their accuracy and error rates.
* Distribution Analysis: Analyzed the prediction distributions to ensure the model's predictions were well-balanced.
* Jaccard Analysis: Applied the Jaccard index to measure the similarity between predicted and actual classifications, aiding in model selection.
  
## **Tools and Technologies**
* Programming Language: Python
* Interactive Programming Tool: Jupyter Notebook
* Data Manipulation and Analysis: Pandas, Numpy
* Data Visualization: Seaborn, Matplotlib
* Data Modeling and Evaluation: Scikit-learn
  
## **Conclusion**
This project demonstrates the effectiveness of Support Vector Machines in cancer detection. The SVM with Polynomial Kernel provided the best results, with high accuracy and reliable performance metrics. The developed model can be a valuable tool in medical diagnostics, contributing to early and accurate detection of cancer.
