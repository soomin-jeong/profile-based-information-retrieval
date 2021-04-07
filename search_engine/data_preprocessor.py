
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def formatize(query: str) -> str:
    query = query.lower()  # lower the capital letters
    query = query.strip()  # remove whitespace
    query = re.sub(r"[0-9]\w+|[0-9]", "", query)
    return query


def tokenize_words(query: str) -> list:
    tokens = word_tokenize(query)
    alphabet_pattern = re.compile('[a-zA-Z]+')
    alpha_only = []
    for each in tokens:
        if alphabet_pattern.match(each) != None:
            alpha_only.append(each)
    return alpha_only


def stemmer(query: list) -> list:
    ps = PorterStemmer()
    return [ps.stem(each) for each in query]


def remove_stopwords(query: list) -> list:
    to_remove = stopwords.words('english')
    return [each for each in query if each not in to_remove]


def preprocess_sentence(sentence: str) -> list:
    sentence = formatize(sentence)
    tokens = tokenize_words(sentence)
    tokens = remove_stopwords(tokens)
    tokens = stemmer(tokens)
    return tokens