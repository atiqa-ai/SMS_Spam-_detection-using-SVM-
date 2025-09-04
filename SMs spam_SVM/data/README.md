## Dataset

The dataset used in this project is an Excel file named `spam.xlsx` (or `.csv`) containing **SMS messages labeled as spam or ham**.  

**Structure of the dataset:**  
- **label**: The category of the message (`spam` or `ham`)  
- **message**: The actual text of the message  

**Example Row:**

| label | message                                      |
|-------|---------------------------------------------|
| ham   | Hey, are we still meeting for lunch today? |
| spam  | Congratulations! You won a free gift card! |

**Dataset source:**  
- Collected from public datasets available for spam detection tasks  
- Uploaded as part of this repository for educational purposes  

**Notes:**  
- The dataset contains **thousands of messages**, making it suitable for training a basic text classification model.  
- It is preprocessed in the project pipeline to remove punctuation, stopwords, and perform stemming before feeding into the SVM classifier.
