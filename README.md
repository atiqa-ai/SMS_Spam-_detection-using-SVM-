
---

## **2️⃣ README (README.md)**


# Spam Detection using SVM

🚀 **Project Overview:**  
This project is a **Spam Classifier** that identifies SMS messages as **spam or ham (not spam)** using Python. It uses **SVM** and **Bag of Words (BoW)** for classification. Perfect for beginners to learn text classification.

---
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)](https://scikit-learn.org/)  
[![Made with](https://img.shields.io/badge/Made%20with-💻%20Machine%20Learning-green)]()  
[![Status](https://img.shields.io/badge/Status-Learning%20Project-yellow)]()  

---

## Dataset

- File: `spam.xlsx`  
- Columns:
  - **label**: `spam` or `ham`  
  - **message**: SMS text  

**Example Row:**

| label | message                                      |
|-------|---------------------------------------------|
| ham   | Hey, are we still meeting for lunch today? |
| spam  | Congratulations! You won a free gift card! |

---

## Model Description

- **Algorithm:** Support Vector Machine (SVM)  
- **Features:** Bag of Words representation of messages  
- **Preprocessing:** Lowercase, remove punctuation, remove stopwords, stemming  
- **Evaluation Metrics:** Accuracy, Classification Report, Confusion Matrix  
- **Optional:** GridSearchCV to optimize parameters  

**Workflow:**  
1. Load dataset  
2. Preprocess text  
3. Convert text to numerical features (BoW)  
4. Split data into train/test sets  
5. Train SVM model  
6. Evaluate performance  
7. Test user input messages  

---

## Plots

1. **Class Distribution** – Spam vs Ham count  
2. **Confusion Matrix** – Model performance  
3. **Word Cloud** – Most frequent words in spam/ham  

---



