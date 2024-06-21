**## Spam Classification with Naive Bayes Classifier**

**Overview**
This project demonstrates a text classification task where we identify spam messages using a Naive Bayes classifier. The dataset used for this project contains SMS messages labeled as 'spam' or 'ham' (non-spam). The goal is to preprocess the text data, train a classifier, and evaluate its performance through various metrics and visualizations.

**Dataset**
The dataset is sourced from a public GitHub repository and contains two columns:

label: The classification label, where 'spam' indicates a spam message and 'ham' indicates a non-spam message.
message: The SMS message text.

**Project Steps**

**1. Data Loading and Preprocessing**

Loading the Data: The dataset is loaded into a Pandas DataFrame.
**Preprocessing:**
Text data is cleaned by:
    Converting to lowercase.
    Removing numbers.
    Removing non-alphabetic characters.
    Removing extra whitespaces.
    Removing stopwords (common words that do not carry significant meaning).

**2. Data Splitting**
The data is split into training and testing sets with an 80-20 split using train_test_split from scikit-learn.

**3. Text Vectorization**
The text data is converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

**4. Model Training**
A Naive Bayes classifier (MultinomialNB) is trained on the TF-IDF transformed training data.

**5. Model Evaluation**

  The classifier's performance is evaluated using:
    Accuracy Score: The ratio of correctly predicted instances to the total instances.
    
    Classification Report: Precision, recall, f1-score, and support for each class.
    
    Confusion Matrix: A matrix showing the counts of true positive, true negative, false positive, and false negative predictions.
    
    ROC Curve: A graph showing the trade-off between the true positive rate and false positive rate.
    
    Feature Importance: Identification of the most important features (words) for predicting spam messages.

**6. Visualizations**

  **Class Distribution:** A count plot showing the distribution of 'ham' and 'spam' messages.
  
  **Confusion Matrix:** A heatmap visualizing the confusion matrix.
  
  **ROC Curve:** A plot of the ROC curve with the AUC score.
  
  **Top Features:** A bar chart showing the top 20 features (words) that are most indicative of spam messages.
