import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textstat import syllable_count
import nltk

# Download stopwords if not already done
nltk.download('stopwords')
nltk.download('punkt')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load positive and negative words dictionaries
positive_words = set(open("positive-words.txt").read().splitlines())
negative_words = set(open("negative-words.txt").read().splitlines())

# Cleaning function
def clean_text(text):
    words = word_tokenize(text.lower())
    return [word for word in words if word.isalpha() and word not in stop_words]

# Function to extract article content
def extract_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1').text.strip()  # Adjust selector for title
    content = ' '.join([p.text.strip() for p in soup.find_all('p')])  # Adjust for content
    return title, content

# Function to compute metrics
def compute_metrics(title, text):
    words = clean_text(text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    complex_words = [word for word in words if syllable_count(word) > 2]

    # Sentiment Analysis
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (word_count + 0.000001)

    # Readability and other metrics
    avg_sentence_length = word_count / len(sentences) if sentences else 0
    percent_complex_words = len(complex_words) / word_count if word_count else 0
    fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count else 0
    syllables_per_word = syllable_count(' '.join(words)) / word_count if word_count else 0
    personal_pronouns = sum(1 for word in words if word.lower() in ["i", "we", "me", "us", "our", "my", "mine", "yours", "you"])

    # Metrics Dictionary
    return {
        "Title": title,
        "Positive Score": positive_score,
        "Negative Score": negative_score,
        "Polarity Score": polarity_score,
        "Subjectivity Score": subjectivity_score,
        "Avg Sentence Length": avg_sentence_length,
        "Percentage of Complex Words": percent_complex_words,
        "FOG Index": fog_index,
        "Avg Words Per Sentence": avg_sentence_length,
        "Complex Word Count": len(complex_words),
        "Word Count": word_count,
        "Syllables Per Word": syllables_per_word,
        "Personal Pronouns": personal_pronouns,
        "Avg Word Length": avg_word_length,
    }

# Main processing
def main(input_file, output_file):
    input_df = pd.read_excel(input_file)
    output_data = []

    for index, row in input_df.iterrows():
        url_id, url = row["URL_ID"], row["URL"]
        try:
            title, content = extract_article(url)
            metrics = compute_metrics(title, content)
            output_data.append({"URL_ID": url_id, "URL": url, **metrics})
        except Exception as e:
            print(f"Error processing URL_ID {url_id}: {e}")

    # Save to output Excel
    output_df = pd.DataFrame(output_data)
    output_df.to_excel(output_file, index=False)

# Execute the program
if __name__ == "__main__":
    main("Input.xlsx", "Output Data Structure.xlsx")
