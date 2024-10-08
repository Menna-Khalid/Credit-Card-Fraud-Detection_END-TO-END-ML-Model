# Credit Card Fraud Detection: End-to-End Machine Learning Model

## 1. Introduction
This project focuses on building an **end-to-end machine learning model** to detect fraudulent credit card transactions using a dataset from **European cardholders**. The dataset is highly imbalanced, with only **492 frauds out of 284,807 transactions**.

### Key Goals:
* Create a model to accurately detect fraud while managing class imbalance.
* Handle preprocessing, feature engineering, model selection, and tuning.
* Deploy the best-performing model for real-time use.

---

## 2. Data Concepts

### 2.1 Data Acquisition and Cleaning
* **Missing Values**: The dataset had no missing values.
* **Duplicates**: 1,081 duplicate rows were removed.
* **Outliers**: Outliers in the `Amount` feature were considered crucial for detecting fraud and were kept.

### 2.2 Exploratory Data Analysis (EDA)
The following key insights were gained from the analysis:
* **Class Imbalance**: The dataset is highly imbalanced with only 0.172% fraud cases.
* **Feature Distributions**: The features `Amount` and `Time` showed specific distributions for fraudulent and non-fraudulent transactions.

### 2.3 Data Transformation and Feature Engineering
* **Log Transformation** was applied to the `Amount` feature.
* **MinMax Scaling** was applied to several important features.
* **Feature Selection**: Feature importance was assessed using **Random Forest**, with key features identified.

---

## 3. Model Concepts

### 3.1 Handling Class Imbalance
* **SMOTE** was used to oversample the minority class to a **60:40 ratio**, ensuring a better balance between fraud and non-fraud cases.

### 3.2 Model Selection
The following models were considered:
* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**
* **Decision Tree**
* **XGBoost** (Final best-performing model)

### 3.3 Hyperparameter Tuning
**GridSearchCV** was used for hyperparameter tuning to find the optimal settings for each model:
* **Decision Tree**
* **Logistic Regression**
* **KNN**

---

## 4. Model Training and Evaluation

### 4.1 Training
Models were trained using the training data split, and evaluation was conducted using metrics like accuracy, precision, recall, and F1 score.

### 4.2 Final Model: XGBoost
* **Accuracy**: 98.28%
* **Precision**: 98.46%
* **Recall**: 96.91%
* **F1 Score**: 97.68%
* **ROC-AUC Score**: 99.8%

These metrics indicate that the model performs very well on the test data, particularly in detecting fraud while managing class imbalance.

### 4.3 Overfitting Issues and Solutions
Initial models like the **Decision Tree** and **XGBoost** showed overfitting with 100% accuracy on the training set. This was addressed by:
* **Regularization**: Increased regularization to reduce overfitting.
* **Hyperparameter Tuning**: Adjusted parameters like `max_depth`, `learning_rate`, and `n_estimators`.

---

## 5. Final Model and Deployment

### 5.1 Saving and Deployment
The **XGBoost model** was saved using `pickle` for deployment, allowing it to be used for real-time fraud detection:
```python
import pickle
with open('Final_Credit_Card_Fraud_Detection_Model.pkl', 'wb') as file:
    pickle.dump(classifiers['XGB Classifier'], file)
```
---

## 5.2 Model Maintenance

### Post-deployment steps include:
   * **Monitoring**: Continuously monitor model performance to ensure it handles new data effectively.
   * **Retraining**: Periodic retraining with new data to ensure the model stays up-to-date with potential changes in fraud patterns.

---

## 6. Lessons Learned

  * **Handling Class Imbalance**: Techniques like SMOTE helped balance the dataset, significantly improving model performance.
  * **Model Overfitting**: Early stopping, regularization, and careful tuning of hyperparameters were key to addressing overfitting.
  * **Importance of Feature Selection**: Identifying important features using Random Forest improved model efficiency and accuracy.

---

## 7. Conclusion

## The XGBoost model successfully detected fraudulent transactions with high accuracy and precision. Proper handling of class imbalance, thoughtful feature engineering, and rigorous tuning were critical to the model's success.
