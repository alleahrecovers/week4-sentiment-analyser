# Sentiment Analyzer — Analysis Report

1. Overall Model Performance

My sentiment analyzer achieved 55% accuracy on the regular test set and 56% accuracy on the challenge set. 
The training dataset contained 50 positive reviews and 50 negative reviews, so the classes were balanced.

The model performed slightly better on the challenge set, but both accuracy scores show that there is plenty of room for improvement.

2. Challenge Category Results

Category      |   Accuracy   |  Correct
Negation            60%           3/5
Sarcasm             20%           1/5
Mixed sentiment     60%           3/5
Intensity           80%           4/5
Domain shift        60%           3/5


Sarcasm was the most difficult category for my model, with only 20% accuracy. The model sometimes focused on positive words without understanding that the actual meaning of the review was negative.
Intensity was the strongest category at 80%.


3. Why the Model Struggled

My model uses a bag-of-words approach with TF-IDF, meaning it looks at which words appear in reviews and how frequently they occur. It does not understand the order of words or the full meaning behind a sentence.

The biggest problem was that this made sarcasm, negation, and mixed sentiment difficult to recognize. For example, the phrases "good" and "not good" contain similar words but have very different meanings. The model does not fully understand that difference.


4. How I Would Improve the Model

One improvement I would make is using bigrams instead of only individual words. Bigrams allow the model to recognize two-word combinations such as "not good" and "very bad." 
This would help preserve some of the meaning that gets lost when looking at individual words.

The downside is that bigrams create more features, which can increase memory usage and training time. 
However, I think this would be worth trying because it could improve the model's understanding of phrases.

I would also expand the training dataset with more examples of sarcasm, negation, and reviews from different subjects.


5. Would I Deploy This Model?

I would not fully trust this model to handle customer reviews on its own yet. 
If a company used it to identify unhappy customers, sarcastic negative reviews could incorrectly be marked as positive.

I would recommend having a human review uncertain predictions and potentially negative customer feedback. This would help prevent frustrated customers from being overlooked. 
The model is a good starting point, but it needs improvement and more testing before being used in a real business environment.
