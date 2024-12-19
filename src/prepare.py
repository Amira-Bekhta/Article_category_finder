from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import torch
from torch.utils.data import DataLoader

stemmer = SnowballStemmer("english")

def tokenize(text):
    return [stemmer.stem(token) for token in word_tokenize(text)]

stopWords = stopwords.words("english")

vectorizer = TfidfVectorizer(
    tokenizer = tokenize,
    stop_words = stopWords,
    max_features = 5000
)

def vectorize(content, fit=True):
    if fit:
        transform = vectorizer.fit_transform(content)
    else:
        transform = vectorizer.transform(content)
    return torch.tensor(transform.toarray()).float()

def build_loader(dataset, test):
    return DataLoader(dataset, batch_size=64, shuffle=(not test))
