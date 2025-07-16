import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already present
nltk.download('vader_lexicon', quiet=True)

# Load your CSV (change 'input.csv' to your filename if needed)
df = pd.read_csv('C:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\Combined Data.csv')

# Ensure there is a 'statement' column
if 'statement' not in df.columns:
    raise ValueError("CSV must have a 'statement' column.")

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return 1
    elif score <= -0.05:
        return -1
    else:
        return 0

# Apply sentiment classification
df['sentiment'] = df['statement'].apply(classify_sentiment)

# Save to output.csv
df.to_csv('output.csv', index=False)
df['sentiment'].value_counts(normalize=True)
print("Sentiment classification complete. Results saved to output.csv.")
