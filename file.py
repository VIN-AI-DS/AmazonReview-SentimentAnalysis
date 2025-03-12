import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr

# Load dataset
data = pd.read_csv("/mnt/data/Reviews.csv")

# Data Cleaning
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

data['cleaned_text'] = data['Text'].apply(clean_text)

# Label Creation (Positive = 1, Negative = 0)
data['label'] = data['Score'].apply(lambda x: 1 if x > 3 else 0)

# TF-IDF Vectorization (Optimized)
vectorizer = TfidfVectorizer(max_features=3000, 
                             sublinear_tf=True, 
                             min_df=5, 
                             max_df=0.9)
X = vectorizer.fit_transform(data['cleaned_text'])  # Sparse matrix format to reduce memory usage

# Target variable (no changes needed here)
y = data['label'].values  # Efficiently stored as a NumPy array


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Gradio Deployment
def sentiment_analysis(review):
    cleaned_review = clean_text(review)
    vectorized_review = vectorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(vectorized_review)
    return "Positive" if prediction[0] == 1 else "Negative"

# Gradio Interface
interface = gr.Interface(fn=sentiment_analysis, 
                         inputs="text", 
                         outputs="text",
                         title="Amazon Review Sentiment Analyzer",
                         description="Enter a product review to predict its sentiment.")

interface.launch()
