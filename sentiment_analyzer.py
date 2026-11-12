
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




reviews = [
    # Positive reviews
    ("I absolutely loved this movie.", "positive"),
    ("The acting was amazing and the story was great.", "positive"),
    ("This was one of the best movies I have seen.", "positive"),
    ("I really enjoyed every minute of it.", "positive"),
    ("The characters were wonderful and interesting.", "positive"),
    ("The movie was entertaining from beginning to end.", "positive"),
    ("I would definitely recommend this movie.", "positive"),
    ("The storyline was creative and exciting.", "positive"),
    ("Everything about this movie was excellent.", "positive"),
    ("The performances were outstanding.", "positive"),
    ("I had a fantastic time watching this.", "positive"),
    ("The ending was beautiful and satisfying.", "positive"),
    ("This movie exceeded my expectations.", "positive"),
    ("The special effects were incredible.", "positive"),
    ("I could watch this movie again and again.", "positive"),
    ("The dialogue was funny and well written.", "positive"),
    ("A brilliant movie with a great cast.", "positive"),
    ("I thought the movie was entertaining.", "positive"),
    ("The director did an excellent job.", "positive"),
    ("This is a wonderful film worth watching.", "positive"),
    ("The music and visuals were beautiful.", "positive"),
    ("I was impressed by the quality of this film.", "positive"),
    ("The story kept me interested the entire time.", "positive"),
    ("A great experience with memorable characters.", "positive"),
    ("I absolutely recommend giving this film a chance.", "positive"),
    ("The movie was fun and enjoyable.", "positive"),
    ("The actors gave incredible performances.", "positive"),
    ("I loved the humor and creativity.", "positive"),
    ("This film was better than I expected.", "positive"),
    ("A truly excellent and entertaining movie.", "positive"),
    ("The pacing was perfect and the story was engaging.", "positive"),
    ("I enjoyed the beautiful cinematography.", "positive"),
    ("The characters felt realistic and relatable.", "positive"),
    ("This movie made me smile.", "positive"),
    ("The plot was interesting and well developed.", "positive"),
    ("I was very happy with this movie.", "positive"),
    ("The film was exciting and full of surprises.", "positive"),
    ("A memorable movie with a fantastic ending.", "positive"),
    ("The cast worked perfectly together.", "positive"),
    ("I thought this was a great film.", "positive"),
    ("The story was heartwarming and enjoyable.", "positive"),
    ("The visuals were stunning.", "positive"),
    ("This movie was worth every minute.", "positive"),
    ("I had a wonderful experience watching it.", "positive"),
    ("The film was creative and entertaining.", "positive"),
    ("I really liked the characters.", "positive"),
    ("An excellent movie with a meaningful story.", "positive"),
    ("The acting made this film special.", "positive"),
    ("I would happily watch this again.", "positive"),
    ("One of my favorite movies.", "positive"),

    # Negative reviews
    ("I absolutely hated this movie.", "negative"),
    ("The acting was terrible and the story was boring.", "negative"),
    ("This was one of the worst movies I have seen.", "negative"),
    ("I did not enjoy any part of it.", "negative"),
    ("The characters were dull and uninteresting.", "negative"),
    ("The movie was boring from beginning to end.", "negative"),
    ("I would not recommend this movie.", "negative"),
    ("The storyline was confusing and disappointing.", "negative"),
    ("Everything about this movie was awful.", "negative"),
    ("The performances were very poor.", "negative"),
    ("I had a terrible time watching this.", "negative"),
    ("The ending was disappointing and frustrating.", "negative"),
    ("This movie failed to meet my expectations.", "negative"),
    ("The special effects looked cheap.", "negative"),
    ("I would never watch this movie again.", "negative"),
    ("The dialogue was awkward and poorly written.", "negative"),
    ("A horrible movie with a weak cast.", "negative"),
    ("I thought the movie was painfully boring.", "negative"),
    ("The director did a terrible job.", "negative"),
    ("This is a bad film not worth watching.", "negative"),
    ("The music and visuals were disappointing.", "negative"),
    ("I was disappointed by the quality of this film.", "negative"),
    ("The story failed to keep me interested.", "negative"),
    ("A frustrating experience with forgettable characters.", "negative"),
    ("I cannot recommend giving this film a chance.", "negative"),
    ("The movie was dull and unpleasant.", "negative"),
    ("The actors gave disappointing performances.", "negative"),
    ("I disliked the humor and lack of creativity.", "negative"),
    ("This film was worse than I expected.", "negative"),
    ("A terrible and unentertaining movie.", "negative"),
    ("The pacing was slow and the story was confusing.", "negative"),
    ("I disliked the cinematography.", "negative"),
    ("The characters felt unrealistic and annoying.", "negative"),
    ("This movie made me frustrated.", "negative"),
    ("The plot was boring and poorly developed.", "negative"),
    ("I was very unhappy with this movie.", "negative"),
    ("The film was predictable and disappointing.", "negative"),
    ("A forgettable movie with a terrible ending.", "negative"),
    ("The cast did not work well together.", "negative"),
    ("I thought this was a horrible film.", "negative"),
    ("The story was dull and uninteresting.", "negative"),
    ("The visuals looked cheap and outdated.", "negative"),
    ("This movie was a waste of time.", "negative"),
    ("I had an unpleasant experience watching it.", "negative"),
    ("The film was poorly made and boring.", "negative"),
    ("I really disliked the characters.", "negative"),
    ("A disappointing movie with a meaningless story.", "negative"),
    ("The acting made this film difficult to watch.", "negative"),
    ("I would not watch this again.", "negative"),
    ("One of my least favorite movies.", "negative"),
]




df = pd.DataFrame(reviews, columns=["review", "sentiment"])

print("Dataset class counts:")
print(df["sentiment"].value_counts())




def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


df["cleaned_review"] = df["review"].apply(clean_text)




X = df["cleaned_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)




vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)




model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# Step 7: Evaluate the model

predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=["negative", "positive"]))

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# Step 8: Explain what the model learns

print("\nWhat the model learns:")
print(
    "The model looks at individual words and how often they appear "
    "in reviews. It uses these word patterns to predict whether a "
    "review is positive or negative."
)
