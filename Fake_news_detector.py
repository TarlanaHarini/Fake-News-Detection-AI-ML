import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load data
real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

# Step 2: Add labels
real["label"] = 1  # Real news
fake["label"] = 0  # Fake news

# Step 3: Combine and shuffle
data = pd.concat([real, fake])
data = data.sample(frac=1).reset_index(drop=True)

# Step 4: Split data
X = data["text"]
y = data["label"]

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")

# Optional: Show confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
