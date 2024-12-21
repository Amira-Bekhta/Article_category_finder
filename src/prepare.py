from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

stemmer = SnowballStemmer("english")

def tokenize(text):
    return [stemmer.stem(token) for token in word_tokenize(text)]

stopWords = stopwords.words("english")



def vectorizer():
    return TfidfVectorizer(
    tokenizer = tokenize,
    stop_words = stopWords,
    max_features = 5000
)

