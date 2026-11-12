
import pandas as pd
from sklearn.metrics import accuracy_score

from sentiment_analyzer import model, vectorizer, clean_text



# Each review has an actual sentiment label and a category.

challenge_reviews = [

    # 3.1 Negation
    ("The movie was not good at all.", "negative", "Negation"),
    ("I did not dislike this movie.", "positive", "Negation"),
    ("The acting was never impressive.", "negative", "Negation"),
    ("I cannot say anything bad about this film.", "positive", "Negation"),
    ("The story was not terrible, but it was not great.", "negative", "Negation"),

    # 3.2 Sarcasm
    ("Oh wonderful, another two hours I will never get back.", "negative", "Sarcasm"),
    ("What a fantastic way to waste my evening.", "negative", "Sarcasm"),
    ("I absolutely loved falling asleep halfway through.", "negative", "Sarcasm"),
    ("Brilliant acting, if the goal was to make everyone bored.", "negative", "Sarcasm"),
    ("Best movie ever, said nobody who watched it.", "negative", "Sarcasm"),

    # 3.3 Mixed sentiment
    ("Beautiful cinematography, but the story was terrible.", "negative", "Mixed"),
    ("The acting was excellent, but the ending was disappointing.", "negative", "Mixed"),
    ("I loved the characters, but the dialogue was awful.", "negative", "Mixed"),
    ("The movie started slowly but ended beautifully.", "positive", "Mixed"),
    ("Great music and visuals, but a very boring plot.", "negative", "Mixed"),

    # 3.4 Intensity
    ("The movie was okay.", "positive", "Intensity"),
    ("The movie was absolutely extraordinary.", "positive", "Intensity"),
    ("The movie was somewhat disappointing.", "negative", "Intensity"),
    ("The movie was unbelievably awful.", "negative", "Intensity"),
    ("The movie was pretty enjoyable.", "positive", "Intensity"),

    # 3.5 Domain shift
    ("The restaurant served an amazing meal.", "positive", "Domain Shift"),
    ("The food was cold and tasted terrible.", "negative", "Domain Shift"),
    ("This phone has excellent battery life.", "positive", "Domain Shift"),
    ("The laptop stopped working after one day.", "negative", "Domain Shift"),
    ("The customer service was friendly and helpful.", "positive", "Domain Shift"),
]




challenge_df = pd.DataFrame(
    challenge_reviews,
    columns=["review", "actual_sentiment", "category"]
)




challenge_df["cleaned_review"] = challenge_df["review"].apply(clean_text)

X_challenge = vectorizer.transform(challenge_df["cleaned_review"])




challenge_df["predicted_sentiment"] = model.predict(X_challenge)




print("\nCHALLENGE SET RESULTS")
print("=" * 60)

print(challenge_df[
    ["category", "review", "actual_sentiment", "predicted_sentiment"]
].to_string(index=False))




print("\nACCURACY BY CATEGORY")
print("=" * 60)

for category in challenge_df["category"].unique():

    category_data = challenge_df[
        challenge_df["category"] == category
    ]

    accuracy = accuracy_score(
        category_data["actual_sentiment"],
        category_data["predicted_sentiment"]
    )

    correct = (
        category_data["actual_sentiment"]
        == category_data["predicted_sentiment"]
    ).sum()

    print(
        f"{category}: {accuracy:.0%} "
        f"({correct}/5 correct)"
    )




overall_accuracy = accuracy_score(
    challenge_df["actual_sentiment"],
    challenge_df["predicted_sentiment"]
)

print("\nOverall Challenge Accuracy:")
print(f"{overall_accuracy:.2%}")
