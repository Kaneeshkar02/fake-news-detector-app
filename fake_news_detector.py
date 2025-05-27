import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle  # Use pickle instead of joblib for saving the model

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load datasets
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

# Add labels
true_df['label'] = 1  # Real news
fake_df['label'] = 0  # Fake news

# Combine
df = pd.concat([true_df, fake_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Combine 'title' and 'text' into one column
df['text'] = df['title'] + ' ' + df['text']

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
df['text'] = df['text'].apply(clean_text)

# Features and Labels
X = df['text']
y = df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test_tfidf)
print("\n🤖 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧠 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save the model and vectorizer using pickle ===

# Save the trained model to a file using pickle
with open('models/fake_news_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the TF-IDF vectorizer to a file using pickle
with open('models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
