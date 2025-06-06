# KaggleChallenge
# 🚢 Titanic Survival Prediction using Machine Learning

![SHAP Summary Plot](![image](https://github.com/user-attachments/assets/dd9703a5-744e-4131-909d-2760e288156c)
)

## 🧩 Problem Statement

The sinking of the RMS Titanic in April 1912 is one of the most tragic maritime disasters in history, resulting in the loss of over 1,500 lives. While the ship was famously deemed “unsinkable,” insufficient lifeboats and chaotic evacuation led to many fatalities. Historical records suggest that certain groups of passengers—based on factors such as gender, age, socio-economic status, and family relationships—had higher chances of survival.

This project focuses on leveraging passenger data from the Titanic to **predict survival outcomes**. Using features such as demographics, ticket class, fare paid, and family aboard, we build a machine learning model to predict whether a passenger survived or not.

---

## 🎯 Goal

- Develop an accurate classification model to predict survival.
- Train on a labeled dataset and evaluate using a test set.
- Use evaluation metrics like **Accuracy**, **F1-Score**, and **ROC-AUC**.
- Interpret the model using SHAP to identify key features that influenced survival.

---
## 📊 Insights by Model

### ✅ CatBoost (Best Performance)
```

Accuracy:     0.96
Precision:    0.96
Recall:       0.95
F1-score:     0.95

```
CatBoost delivered exceptional results, outperforming all other models. It achieved **97% recall for class 0** and **95% recall for class 1**, making it highly suitable for production use where precision and recall are both critical.

---

### 🌲 Random Forest
```

Accuracy:     0.84
Precision:    0.85 (class 0), 0.82 (class 1)
Recall:       0.88 (class 0), 0.78 (class 1)
F1-score:     0.86 (class 0), 0.80 (class 1)

```
Random Forest showed balanced and solid performance. It’s reliable, interpretable, and a good alternative when speed or simplicity is preferred.

---

### 🌱 Gradient Boosting
```

Accuracy:     0.84
Precision:    0.87 (class 0), 0.81 (class 1)
Recall:       0.87 (class 0), 0.81 (class 1)
F1-score:     0.87 (class 0), 0.81 (class 1)

```
Gradient Boosting was nearly on par with Random Forest. Its consistency in precision, recall, and F1-score makes it a strong candidate for real-world deployment.

---

### 📉 Logistic Regression
```

Accuracy:     0.80
Precision:    0.82 (class 0), 0.78 (class 1)
Recall:       0.86 (class 0), 0.73 (class 1)
F1-score:     0.84 (class 0), 0.76 (class 1)

```
A good baseline model. Logistic Regression is easy to understand and implement, ideal for scenarios where transparency is required.

---

### ⚠️ Support Vector Machine (SVM)
```

Accuracy:     0.67
Precision:    0.65 (class 0), 0.78 (class 1)
Recall:       0.94 (class 0), 0.28 (class 1)
F1-score:     0.77 (class 0), 0.42 (class 1)

```
SVM had high precision but poor recall for class 1, leading to significant misclassifications. It is biased towards the majority class and **not recommended** without tuning or class balancing techniques.

---

## 📌 Model Recommendation Summary

| Model                | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Notes                                         |
|---------------------|----------|----------------------|------------------|--------------------|-----------------------------------------------|
| 🥇 CatBoost          | 0.96     | 0.96                 | 0.95             | 0.95               | Best performer across all metrics             |
| 🌲 Random Forest     | 0.84     | 0.82                 | 0.78             | 0.80               | Balanced, interpretable, and fast             |
| 🌱 Gradient Boosting | 0.84     | 0.81                 | 0.81             | 0.81               | Consistent and solid performer                |
| 📉 Logistic Regression | 0.80   | 0.78                 | 0.73             | 0.76               | Great as a simple, interpretable baseline     |
| ⚠️ SVM               | 0.67     | 0.78                 | 0.28             | 0.42               | Poor recall, not suited without improvements  |

---

## 📌 Recommendation

- **Use CatBoost** for optimal performance on all metrics.
- Choose **Random Forest or Gradient Boosting** when model simplicity or speed is preferred.
- **Avoid SVM** in its default form due to poor recall unless significant improvements are made through tuning and class balancing.

---

## 📎 Feature Importance (SHAP Summary)

The SHAP plot above visualizes the impact of each feature on the model’s output:
- **Sex**, **Pclass**, and **Age** are the most influential features.
- High SHAP values indicate strong influence in predicting survival.
- Feature coloring indicates feature value (red = high, blue = low).

---

## 📂 Dataset

- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- Train/test split used for supervised learning.

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, LightGBM, CatBoost
- SHAP for explainability
- Jupyter Notebooks

---

## ✅ Evaluation Metrics

- Accuracy
- Precision / Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- SHAP Interpretability

---

## 📬 Contact

For questions, suggestions, or collaboration, feel free to open an issue or contact me via GitHub.
