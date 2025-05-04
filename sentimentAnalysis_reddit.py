

# ------------------------
# Part 1: Import Libraries
# ------------------------
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from summa import summarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# -------------------------------
# Part 2: Analyze Original Dataset
# -------------------------------
# Load dataset
df = pd.read_csv('reddit_realestate_100posts.csv')



# Preprocess the text
def preprocess(text_input):
    text_input = text_input.lower()
    text_input = re.sub(r"http\S+|www\S+", "", text_input)
    text_input = re.sub(r"[^a-zA-Z\s]", "", text_input)
    text_input = re.sub(r"\s+", " ", text_input).strip()
    
    tokens = word_tokenize(text_input)
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w not in stop_words]
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    cleaned = [lemmatizer.lemmatize(stemmer.stem(word)) for word in filtered]
    
    return " ".join(cleaned)

df['clean_text'] = df['Body'].astype(str).apply(preprocess)

# Generate summaries
def summarize_text(text):
    return summarizer.summarize(text, ratio=0.2)

df['summary'] = df['Body'].astype(str).apply(summarize_text)

# Sentiment using VADER
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['Body'].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])

def get_sentiment_label(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['sentiment_score'].apply(get_sentiment_label)

# Importance score
def calculate_importance(row):
    summary_length = len(row['summary'].split())
    normalized_len = summary_length / 100
    return round((abs(row['sentiment_score']) + normalized_len) * 5, 2)

df['importance_score'] = df.apply(calculate_importance, axis=1)
df['normalized_importance'] = df['importance_score'] / df['importance_score'].max()

# ---------------------------------------
# Part 3: CountVectorizer and TF-IDF (Top 10 Words)
# ---------------------------------------
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(df['clean_text'])
count_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# ---------------------------------------
# Part 4: Visualizations
# ---------------------------------------

# Sentiment distribution
plt.figure(figsize=(8, 5))
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
plt.title("Sentiment Distribution (VADER)", fontsize=14)
plt.xlabel("Sentiment Category")
plt.ylabel("Post Count")
plt.tight_layout()
plt.savefig("vader_sentiment_distribution.png")
plt.show()

# Score distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['sentiment_score'], kde=True, color='orange')
plt.title("VADER Sentiment Score Distribution")
plt.xlabel("Compound Score")
plt.tight_layout()
plt.savefig("vader_score_distribution.png")
plt.show()

# CountVectorizer: Top 10 words
plt.figure(figsize=(10, 5))
count_df.sum().nlargest(10).plot(kind='bar', color='purple')
plt.title("Top 10 Words - CountVectorizer")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("top10_countvectorizer.png")
plt.show()

# TfidfVectorizer: Top 10 words
plt.figure(figsize=(10, 5))
tfidf_df.sum().nlargest(10).plot(kind='bar', color='teal')
plt.title("Top 10 Words - TfidfVectorizer")
plt.xlabel("Words")
plt.ylabel("TF-IDF Score")
plt.tight_layout()
plt.savefig("top10_tfidfvectorizer.png")
plt.show()


# ------------------------------
# Part 5: Neat Display Using Rich for tabulated format 
# ------------------------------
from rich.console import Console
from rich.table import Table

# Create console instance
console = Console()

# Create the rich table with nice formatting
table = Table(title="📊 Top 10 Analyzed Reddit Real Estate Posts")

# Define the columns
table.add_column("Title", justify="left", style="bold cyan", no_wrap=True)
table.add_column("Summary", justify="left", style="magenta")
table.add_column("Sentiment", justify="center", style="green")
table.add_column("Importance", justify="center", style="yellow")
table.add_column("Normalized", justify="center", style="white")

# Fill the table with the top 10 results due to nice displaying I didn't display 100 data here
for _, row in df[['Title', 'summary', 'sentiment', 'importance_score', 'normalized_importance']].head(10).iterrows():
    table.add_row(
        str(row['Title'])[:40] + '...' if len(str(row['Title'])) > 40 else str(row['Title']),
        str(row['summary'])[:80] + '...' if len(str(row['summary'])) > 80 else str(row['summary']),
        row['sentiment'],
        f"{row['importance_score']:.2f}",
        f"{row['normalized_importance']:.2f}"
    )

# Print the rich table to terminal
console.print(table)

##You can see the output result as csv format called "analysis_output"

