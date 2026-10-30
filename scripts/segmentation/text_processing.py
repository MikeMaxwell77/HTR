"""Sentence assembly, spelling correction, and original overlap counts."""
import re


def assemble_sentence(predicted_texts):
    # Original IAM form convention: first three entries are header metadata;
    # the final entry is the Name field. Slicing also handles short predictions.
    return ' '.join(word for word in predicted_texts[3:-1] if word not in {'.', '"', ','})


def clean_text(text):
    from textblob import TextBlob
    return str(TextBlob(re.sub(r'\s+', ' ', text)).correct())


def compare_strings(str1, str2):
    return len(set(str1.lower().split()).intersection(str2.lower().split()))
