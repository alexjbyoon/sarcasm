Problem Statement
Sarcasm is one of the hardest linguistic phenomena for NLP models to detect because it relies on context, tone, and cultural cues absent from raw text. Misclassifying sarcastic comments as sincere (or vice versa) harms downstream sentiment analysis, moderation, and recommendation systems. This project trains a classifier to predict whether a Reddit comment is sarcastic.

Dataset Source
Reddit comment dataset in TSV format (sarcasm.train.tsv.gz, sarcasm.test-bal.tsv.gz). The test set is balanced across sarcastic and non-sarcastic labels to ensure fair evaluation.

Approach

Text preprocessing: tokenization, lowercasing, removal of Reddit-specific markup.
Feature extraction using TF-IDF and/or word embeddings.
Classification using machine learning models (logistic regression, Naive Bayes, and/or neural approaches).
Evaluation on the balanced test set to measure precision, recall, and F1-score.

Key Findings

Lexical features alone achieve reasonable baseline accuracy, but context (e.g., parent comment) meaningfully improves performance.
Sarcastic comments tend to use more extreme sentiment words paired with incongruent context.
The balanced test set prevents inflated accuracy from class imbalance.
