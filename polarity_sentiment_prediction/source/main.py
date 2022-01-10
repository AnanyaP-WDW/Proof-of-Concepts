#### main.py

import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
import nltk
from source import scraper

from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings('ignore')

# Load the train data
train_data = pd.read_csv('train.csv')
train_data = train_data.groupby('aspect_term').filter(lambda x: len(x) > 10)
print(f'data_shape: {train_data.shape}')
# shuffle the dataframe
train_data = train_data.sample(frac=1, random_state=77).reset_index(drop=True)

# Text Cleaning
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;:$!]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_?&]')
STOPWORDS = set(stopwords.words('english'))
single_chars = re.compile(r'\s+[a-zA-Z]\s+')

def clean_text(text: str)-> str:
    """
    Preprocesses text and returns a cleaned
    piece of text with unwanted characters removed

    Args:
       text: a string
    Returns:
        Preprocessed text
    """
    text = str(text).lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwords from text
    text = single_chars.sub('', text) #remove single-characters
    return text


def remove_URL(text: str)-> str:
    """
    Removes URL patterns from text.
    Args:
        `text`: A string, word/sentence
    Returns:
        Text without url patterns.
    """
    url = re.compile('https?://\S+|www\.\S+')
    text = url.sub('',text)
    return text


def extract_entity(text):
    """Extract entity name from aspect category."""
    attribute = re.compile('#[a-zA-Z0-9]+')
    underscore = re.compile('_[a-zA-Z0-9]+')
    entity = attribute.sub('', str(text).lower())
    entity = underscore.sub('', entity)

    return entity


def count_vectorize(data: str) -> [int]:
    """
    Create word vectors/tokens for input data
    Args:
        `data`: text to be vectorized
    """
    vectorizer = CountVectorizer(min_df=3, analyzer='word',
                                 ngram_range=(1, 3))
    vectors = vectorizer.fit_transform(data)

    return vectors, vectorizer


def rmse(y_true, y_pred):
    """Computing RMSE metric."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_dir(dir_name: str):
    """Creates a new directory in the current working directory."""
    save_dir = os.path.join('./', dir_name)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    else:
        print(f'{save_dir}: Already exists!')
    return save_dir


# Splitting data into train and test sets
train_df = train_data.iloc[:2100]
test_df = train_data.iloc[len(train_df):]

train_df['text'] = train_df['text'].apply(clean_text)
train_df['text'] = train_df['text'].apply(remove_URL)

test_df['text'] = test_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(remove_URL)

train_df['aspect_term'] = train_df['aspect_term'].apply(clean_text)
test_df['aspect_term'] = test_df['aspect_term'].apply(clean_text)

print(f"\nsentiment tweet distribution: {train_df['polarity'].value_counts()}\n")

#Test-labels
polarity_labels_test = pd.get_dummies(test_df['polarity'].astype(str), dtype='int32')
aspect_labels_test = pd.get_dummies(test_df['aspect_term'], dtype='int32')

# Categorizing polarity and aspect_term labels (multi-label formulation)
# Train
train_df['polarity'] = train_df['polarity'].astype(str)

#remove nan class
train_df = train_df[train_df['aspect_term'] != 'nan']
test_df = test_df[test_df['aspect_term'] != 'nan']

# Class label Encoding
aspect_encoder = LabelEncoder()
train_df['aspect_term'] = aspect_encoder.fit_transform(train_df['aspect_term'])

polarity_encoder = LabelEncoder()
train_df['polarity'] = polarity_encoder.fit_transform(train_df['polarity'])


aspect_labels_train = pd.get_dummies(train_df['aspect_term'], dtype='int32')


print(f"\nNumber of unique aspect_terms: {train_df['aspect_term'].nunique()}\n")

print('\n========================Text Preprocessing=============================\n')

# Create train vectors
train_vectors, count_vectorizer = count_vectorize(train_df['text'])

## Map the tokens in the train vectors to the test set.
# i.e.the train and test vectors use the same set of tokens
test_vectors = count_vectorizer.transform(test_df['text'])

# split data into features and labels for training
y_aspect = train_df['aspect_term'] #aspect_term target

y_polarity = train_df['polarity'] #polarity target

# Building the models

def build_models():
    """Define ABSA ensemble predicition models.
    Returns:
        Absa_model: Aspect prediction model
        Polarity_model: Sentiment prediction model
    """
    absa_model = xgb.XGBRegressor(max_depth=5,
                              n_estimators=150,
                              classes=list(aspect_encoder.classes_),
                              colsample_bytree=0.9,
                              num_class=79,
                              objective='multi:softprob',
                              metric='auc',
                              nthread=2,
                              learning_rate=0.1,
                              random_state=77
                              )

    polarity_model = xgb.XGBRegressor(max_depth=5,
                              n_estimators=350,
                              classes=list(polarity_encoder.classes_),
                              colsample_bytree=0.9,
                              num_class=3,
                              objective='multi:softprob',
                              metric='auc',
                              nthread=2,
                              learning_rate=0.1,
                              random_state=77
                              )
    return absa_model, polarity_model

# training ABSA models
# Using stratifiedKfold to try balancing classes during training.
# This ensures that all classes are almost equally represented in the
## training data thus eliminating bias towards one.

def train_absa_model(train_data, target, model, task='polarity'):
    """Train absa model.
    Args:
        word_vectors: train vectors.
        target: Variable to be predicted (label).
        model: predictive model.
    """
    print('===========================Training ABSA model=======================\n')
    stratified_kf = StratifiedKFold(8, shuffle=True, random_state=77)
    labels = target
    p_scores = []

    for i, (tr, val) in enumerate(stratified_kf.split(train_data, labels), 1):
        X_train, y_train = train_data[tr], np.take(labels, tr, axis=0)
        X_val, y_val = train_data[val], np.take(labels, val, axis=0)

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        score = log_loss(y_val, val_preds)
        p_scores.append(score)
        print(f'Fold-{i} log_loss: {score}')
    print(f'Mean_log_loss: {np.mean(p_scores)}')
    print('\n===============Saving trained model=============================\n')

    save_dir = make_dir('saved_models')
    # Save model
    if task=='absa':
        pickle.dump(model, open(save_dir+'/absa_model.pkl', 'wb'))
    else:
        pickle.dump(model, open(save_dir+'/polarity_model.pkl', 'wb'))

    return model


# Making predictions
def predict_on_test(test_data, model, task='absa'):
    """Make predictions on test data set with
    the model.
    Args:
        test_data: test vectors
        model: trained model
        type: polarity or absa predictions
    """
    predictions = model.predict(test_data)
    print(f'Predictions_shape: {predictions.shape}')

    #Test-data performance scores
    test_scores = []
    if task == 'polarity':
        for i in range(len(polarity_encoder.classes_)):
            loss =     log_loss(polarity_labels_test[polarity_encoder.classes_[i]], predictions[:,i])
            test_scores.append(loss)
        print(f'Mean Test_loss_{task}: {np.mean(test_scores)}')
    else:
        #print(f'Aspect classes: {aspect_encoder.classes_}')
        for i in range(len(aspect_encoder.classes_)):
            loss =     log_loss(aspect_labels_test[aspect_encoder.classes_[i]], predictions[:,i])
            test_scores.append(loss)
        print(f'Mean Test_loss_{task}: {np.mean(test_scores)}')

def inverse_predict(data: [str], tokenizer, model2):
    """Reverse numerical predictions for each word in a list from numeric
    value to text.
    """
    result_list = list()
    top_aspect_term_list = list()
    wordcloud_string = ''
    import yake
    extractor = yake.KeywordExtractor(lan='en',
                                      n=1,
                                      top=5
                                      )
    data = np.array(data)
    text = data.flatten().tolist()
    for i, c in enumerate(data):
        word_to_vec = tokenizer.transform(c)
        aspects = extractor.extract_keywords(text[i])
        # bar plots
        #plt.bar(dict(list(aspects)).keys(), dict(list(aspects)).values())
        # Create the wordcloud object
        #wordcloud = WordCloud(width=480, height=480, margin=0).generate(' '.join(list(dict(list(aspects)).keys())))
        # Display the generated image:
        #plt.imshow(wordcloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.margins(x=0, y=0)
        #plt.savefig(f'./wcloud_{i}.png')
        #plt.show()
        top_aspect_term = list(aspects)[-1][0]
        top_aspect = list(aspects)[-1][1]
        top_aspect_term_list.append(top_aspect)

        #predictions = aspect_encoder.inverse_transform(np.argmax(model1.predict(word_to_vec), 1))
        polarity_pred = polarity_encoder.inverse_transform(np.argmax(model2.predict(word_to_vec),1))

        result_list.append(f"The Review: {str(c[0])}_\n is expressing a {str(polarity_pred[0])} sentiment about {top_aspect_term}")

    wordcloud_string += " ".join(top_aspect_term_list)+" "
    wordcloud = WordCloud(width=480, height=480, margin=0).generate(wordcloud_string)
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title("Word Cloud of top aspects")
    #plt.imshow(wordcloud, interpolation='bilinear')


    return result_list[0:10],wordcloud


def get_data(search_word):
    """Run absa models on unseen data"""
    #import scraper
    test_data, _ = scraper.get_twitter_data(search_word)
    return test_data

#
# # if __name__=='__main__':
# test_data = get_data("rice")
# aspect_model, sentiment_model = build_models()
#   #absa_model = train_absa_model(train_vectors, y_aspect, aspect_model)
#
# polarity_model = train_absa_model(train_vectors, y_polarity, sentiment_model, task='polarity')
#   #print('==============Running sentiment predictions=================')
# predict_on_test(test_vectors, polarity_model, task='polarity')
#
#   # Run model against scraped twitter data
# result_1, plot_1,abc = inverse_predict(test_data, count_vectorizer, polarity_model)
#
# abc
