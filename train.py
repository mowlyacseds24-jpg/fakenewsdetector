import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

# Combine
df = pd.concat([fake, true])

# IMPORTANT: merge text properly
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

# Shuffle (VERY IMPORTANT)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Check accuracy
print("Accuracy:", model.score(X_test_vec, y_test))

# Save
joblib.dump(model, "lr_model.jb")
joblib.dump(vectorizer, "vectorizer.jb")