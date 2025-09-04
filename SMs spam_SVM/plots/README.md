## Plots

Visualizations in this project help in **understanding the data and evaluating the model**.  

**Included Plots:**  

1. **Class Distribution:**  
   Shows the number of spam vs ham messages in the dataset. Helps to understand dataset balance.  

2. **Confusion Matrix:**  
   Displays **predicted vs actual labels**, giving a clear view of model performance on test data.  
**Example of how plots are generated in the notebook:**
```python
# Class Distribution
df['label'].value_counts().plot(kind='bar', color=['skyblue','salmon'])

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=grid.classes_).plot()
