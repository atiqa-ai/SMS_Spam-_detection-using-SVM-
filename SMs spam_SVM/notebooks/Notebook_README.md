## Jupyter Notebook

The core of this project is a **Jupyter Notebook** (`spam_classifier.ipynb`) which walks you through the **entire workflow** of building a spam detection model using **SVM**.  

**Notebook Highlights:**  
1. **Data Loading:** Loads the Excel dataset (`spam.xlsx`) with SMS messages labeled as spam or ham.  
2. **Text Preprocessing:**  
   - Converts text to lowercase  
   - Removes punctuation  
   - Removes stopwords  
   - Performs stemming using PorterStemmer  
3. **Feature Extraction:** Converts messages into numerical features using **Bag of Words (BoW)**.  
4. **Train-Test Split:** Splits dataset into training and testing sets.  
5. **Model Training:** Trains an **SVM classifier** and optionally uses **GridSearchCV** to find the best hyperparameters.  
6. **Evaluation:** Computes accuracy and prints classification report.  
7. **User Input Testing:** Allows testing new messages for spam detection in real-time.  

The notebook is **step-by-step**, making it easy for beginners to understand **text classification using Python**.

---
