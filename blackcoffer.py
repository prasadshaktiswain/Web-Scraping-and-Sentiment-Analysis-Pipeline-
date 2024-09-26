import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from syllapy import count as syllables_count
import os
from nltk.corpus import cmudict
from nltk.corpus import stopwords
import string
import re


def read_excel_file():
    url="/home/shakti/Desktop/project2/balck_coffer/input.xlsx"
    df=pd.read_excel(url)
    return df

def process_dataframe(df):   
    data = []
    for i,row in df.iterrows():
        url = row['URL']  
        url_id = row['URL_ID']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find("h1", {"class": "entry-title"})
        body_tag = soup.find("div", {"class": "td-ss-main-content"})
        title = title_tag.get_text() if title_tag else ''
        body = body_tag.get_text() if body_tag else ''
        article = title +'\n'+ body
        data.append(article)
    return str(data)

stopword_path="/home/shakti/Desktop/project2/balck_coffer/StopWords"
def read_stop_words(stopword_path):
    files_contents = {}
    for filename in os.listdir(stopword_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(stopword_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                files_contents[filename] = f.read()
    return files_contents

# nltk.download("stopwords")
def remove_stopwords(text,stop_words):
    word_tokens = word_tokenize(text)
    return [word for word in word_tokens if word.casefold() not in stop_words]

positive_word_path="/home/shakti/Desktop/project2/balck_coffer/positive-words.txt"
negative_word_path="/home/shakti/Desktop/project2/balck_coffer/negative-words.txt"

def load_master_dictionary(positive_word_path, negative_word_path):
    positive_words = {}
    negative_words = {}
    with open(positive_word_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word = line.strip() 
            positive_words[word] = 1
    with open(negative_word_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word = line.strip()  
            negative_words[word] = -1
    return positive_words, negative_words    

def analyze_sentiment(text, positive_words, negative_words):
    positive_count = sum(word in positive_words for word in text)
    negative_count = sum(word in negative_words for word in text)
    
    if positive_count > negative_count:
        return 'Positive'
    elif positive_count < negative_count:
        return 'Negative'
    else:
        return 'Neutral'
    
def calculate_scores(text, positive_words, negative_words):
    tokens = nltk.word_tokenize(text)
    positive_score = sum(word in positive_words for word in tokens)
    negative_score = sum(word in negative_words for word in tokens)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

def calculate_readability(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    complex_words = [word for word in words if len(word) > 2] 

    average_sentence_length = len(words) / len(sentences)
    percentage_complex_words = len(complex_words) / len(words) * 100
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)

    return average_sentence_length, percentage_complex_words, fog_index

def calculate_average_words_per_sentence(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    average_words_per_sentence = len(words) / len(sentences)
    return average_words_per_sentence

def count_complex_words(text):
    words = text.split()
    complex_word_count = sum(1 for word in words if syllables_count(word) > 2)
    return complex_word_count

def calculate_cleaned_word_count(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return len(cleaned_words)

def count_syllables_per_word(text):
    words = text.split()
    syllables_per_word = []
    for word in words:
        if word.endswith(("es", "ed")):
            syllables_per_word.append(1)
            continue
        vowels = "aeiouAEIOU"
        count = 0
        for index in range(len(word)):
            if word[index] in vowels and (index == 0 or word[index - 1] not in vowels):
                count += 1
        if word.endswith("e") and count > 1:
            count -= 1
        syllables_per_word.append(count)
    return syllables_per_word

def calculate_personal_pronouns(text):
    personal_pronouns = re.findall(r'\bI\b|\bwe\b|\bmy\b|\bours\b|\bus\b', text, re.IGNORECASE)
    personal_pronouns = [pronoun for pronoun in personal_pronouns if pronoun.lower() != 'us' or re.search(r'\b[Uu][Ss]\b', text) is None]
    return len(personal_pronouns)

def calculate_average_word_length(text):
    words = nltk.word_tokenize(text)
    average_word_length = sum(len(word) for word in words) / len(words)
    return average_word_length

def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['text', 'Sentiment', 'Positive Score', 'Negative Score', 'Polarity Score', 'Subjectivity Score', 'Average Sentence Length', 'Percentage Complex Words', 'Fog Index', 'Average Words per Sentence', 'Complex Word Count', 'Cleaned Word Count', 'Personal Pronouns Count', 'Average Word Length'])
    df.to_csv(filename, index=False)

# function calls
df = read_excel_file()
text=process_dataframe(df)
# print(text)
stop_words=read_stop_words(stopword_path)
# print(stop_words)
cleaned_text = remove_stopwords(text,stop_words)
# print(cleaned_text)
positive_words,negative_words=load_master_dictionary(positive_word_path,negative_word_path)
print(positive_words,negative_words)
sentiment = analyze_sentiment(cleaned_text, positive_words, negative_words)
print(sentiment)
positive_score, negative_score, polarity_score, subjectivity_score = calculate_scores(text, positive_words, negative_words)
print(positive_score, negative_score, polarity_score, subjectivity_score)
average_sentence_length, percentage_complex_words, fog_index = calculate_readability(text)
print(average_sentence_length, percentage_complex_words, fog_index)
average_words_per_sentence = calculate_average_words_per_sentence(text)
print(average_words_per_sentence)
complex_word_count = count_complex_words(text)
print(complex_word_count)
cleaned_word_count = calculate_cleaned_word_count(text)
print(cleaned_word_count)
syllables_per_word = count_syllables_per_word(text)
# print(syllables_per_word)
personal_pronouns_count = calculate_personal_pronouns(text)
print(personal_pronouns_count)
average_word_length = calculate_average_word_length(text)
print(average_word_length)
data = [(text, sentiment, positive_score, negative_score, polarity_score, subjectivity_score, average_sentence_length, percentage_complex_words, fog_index, average_words_per_sentence, complex_word_count, cleaned_word_count, personal_pronouns_count, average_word_length)]
save_to_csv(data, 'output.csv')