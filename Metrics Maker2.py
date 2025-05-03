# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:47:32 2025

@author: mikey
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
"""
Sources:
    https://spencerporter2.medium.com/understanding-cosine-similarity-and-word-embeddings-dbf19362a3c
    https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python
"""
import re

def basic_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())
#import data
predictions_df = pd.read_csv("model_predictions_chatGPT_responses.csv")
#preprocess texts
def preprocess(text):
    # Convert to lowercase and tokenize
    tokens = basic_tokenize(text)

    # Remove punctuation and convert back to string
    words = [word for word in tokens if word.isalnum()]
    return ' '.join(words)
#Cosine Similarity
def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def word_overlap_percentage(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    overlap = words1.intersection(words2)
    return len(overlap) / ((len(words1) + len(words2)) / 2)

vector_victor = []
word_overlap_record = []
for index, row in predictions_df.iterrows():
    
    text1 = row['Chat GPT']
    text2 = row['text_true']

    text1_processed = preprocess(text1)
    text2_processed = preprocess(text2)

    print("Preprocessed Text 1:", text1_processed)
    print("Preprocessed Text 2:", text2_processed)

    cosine_sim = calculate_cosine_similarity(text1_processed, text2_processed)
    vector_victor.append(cosine_sim)
    print(f"\n1. Cosine Similarity: {cosine_sim:.4f}")
    
    word_overlap = word_overlap_percentage(text1_processed, text2_processed)
    word_overlap_record.append(word_overlap)
    print(f"2. Word Overlap Percentage: {word_overlap:.4f}")
    
#get average
print(f"Average Cosine Similarity Score: {sum(vector_victor)/len(vector_victor):.4f}")
print(f"Average Word Overlap Percentage: {sum(word_overlap_record)/len(word_overlap_record):.4f}")