import os
import re
import string
import nltk
import pandas as pd
import unicodedata
from nltk import WordNetLemmatizer,  word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')


def preprocess_text(text):
    # Unicode Normalization
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Lower word
    text = text.lower()

    # Remove url and html tag
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation and special character
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    # Correct lemmatize with post tag
    pos_tags = nltk.pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) for word, pos_tag in zip(tokens, pos_tags)]

    # Remove short words
    tokens = [word for word in tokens if len(word) > 2]

    # Clear space in word
    text = ' '.join(tokens)
    text = re.sub('\s+', ' ', text).strip()

    return text

def get_wordnet_pos(pos_tag):
    tag = pos_tag[1][0].upper()
    tag_dict = {'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_data(df):
    # Convert text to TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    df['Text'] = df['Text'].apply(preprocess_text)
    X = vectorizer.fit_transform(df['Text']).toarray()
    y = df['oh_label']

    return X, y, vectorizer


def load_datasets():
    all_df = []
    for file in os.listdir('data/'):
        df = pd.read_csv('data/' + file, usecols=['Text', 'oh_label'])
        all_df.append(df)

    dataset = pd.concat(all_df, ignore_index=True)
    dataset.dropna(subset=['oh_label'], inplace=True)
    return dataset