
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset from the web
news_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_or_real_news.csv'
news_df = pd.read_csv(news_url)

print("Sample entries from the dataset:")
print(news_df[['title', 'label']].head(), "\n")

# Splitting features and labels
content = news_df['text']
target = news_df['label']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    content, target, test_size=0.2, random_state=0
)

# Applying TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)
X_test_vectorized = tfidf_vectorizer.transform(X_test)

# Training the model
classifier = PassiveAggressiveClassifier(max_iter=100)
classifier.fit(X_train_vectorized, y_train)

# Predicting and evaluating
predictions = classifier.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"🔍 Prediction Accuracy: {round(accuracy * 100, 2)}%")
print("📊 Confusion Matrix:")
print(conf_matrix)
