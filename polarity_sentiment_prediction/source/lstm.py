### lstm.py
import os
import re
import numpy as np 
import pandas as pd
import tensorflow as tf
import warnings
import nltk

from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss, mean_squared_error
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

warnings.filterwarnings('ignore')
nltk.download('stopwords')

# Load the train data
train_data = pd.read_csv('./train.csv')
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


# Splitting data into train and test sets
train_df = train_data.iloc[:2900]
test_df = train_data.iloc[len(train_df):]

train_df['text'] = train_df['text'].apply(clean_text)
train_df['text'] = train_df['text'].apply(remove_URL)

test_df['text'] = test_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(remove_URL)



# Categorizing polarity and aspect_term labels (multi-label formulation)
# Train
train_df['polarity'] = train_df['polarity'].astype(str)

#remove nan class
train_df = train_df[train_df['aspect_term'] != 'nan']
test_df = test_df[test_df['aspect_term'] != 'nan']


polarity_encoder = LabelEncoder()
train_df['polarity'] = polarity_encoder.fit_transform(train_df['polarity'])

#Test-labels
polarity_labels_test = pd.get_dummies(test_df['polarity'], dtype='int32')

print('\n========================Text Preprocessing=============================\n')
# text lengths
token_length = [len(x.split(" ")) for x in train_df['text']]
print(f'Max_token length: {max(token_length)}\n')


# split data into features and labels for training
X = train_df['text']
y_polarity = train_df['polarity'] #polarity target

# Text prepocessing
train_corpus = X.tolist()
test_corpus = test_df['text'].tolist()

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_corpus)
vocab_size = len(tokenizer.word_counts)
print(f'\nTrain Vocabulary size: {vocab_size}\n')

# Sequence lengths (vocabulary size in a given sequence)
# Computing the vocabulary size per percentile
print('===========Analyzing the vocabulary size per percentile==============')
seq_lengths = np.array([len(s.split()) for s in train_corpus])
print(f'{[(p, np.percentile(seq_lengths, p)) for p in [75, 80, 90, 95, 99, 100]]}')

max_seqlen = 64

# Train encodings (words/sentences >> int) with padding
# Padding ensures that sequences are of the same length
train_encodings = tokenizer.texts_to_sequences(train_corpus)
train_encodings = tf.keras.preprocessing.sequence.pad_sequences(
    train_encodings, maxlen = max_seqlen)
polarity_labels = np.array(y_polarity)

# Creating a train dataset
aspect_dataset = tf.data.Dataset.from_tensor_slices(
    (train_encodings, polarity_labels))

# Test encodings with padding
test_encodings = tokenizer.texts_to_sequences(test_corpus)
test_encodings = tf.keras.preprocessing.sequence.pad_sequences(
    test_encodings, maxlen= max_seqlen)
test_labels = np.zeros_like(polarity_labels_test) # Predictions placeholder

# Test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(
 (test_encodings, test_labels))

def encode(text):
    text_encodings = tokenizer.texts_to_sequences(text)
    text_encodings = tf.keras.preprocessing.sequence.pad_sequences(
    text_encodings, maxlen= max_seqlen)
    dataset = (tf.data.Dataset.from_tensor_slices((text_encodings)))
    return dataset

# Creating train and test batches

# Train-validation split and batch creation
aspect_dataset = aspect_dataset.shuffle(2000)

val_size = (len(train_corpus)) // 7
val_dataset_absa = aspect_dataset.take(val_size)

aspect_dataset = aspect_dataset.skip(val_size)

batch_size = 8
aspect_dataset = aspect_dataset.batch(batch_size).repeat()

val_dataset_absa = val_dataset_absa.batch(batch_size)
print(f'Validation_size: {val_size}')
print(f'Train_size: {len(train_corpus)}')

# test batch creation
test_batched = test_dataset.batch(batch_size)


# Building the model
# 
# The model consist of:
#      * An Embedding layer (to generate word embeddings)
#      A Bidirectional LSTM layer
#      1 hidden Dense layers with the `relu` activation function
#      An output Dense layer


def rmse(y_true, y_pred):
    """Computing the RMSE metric."""
    return K.sqrt(K.mean((K.square(y_pred - y_true))))



embedding_dim=64

# ABSA model
absa_model = tf.keras.Sequential([
    layers.Embedding(vocab_size+1, embedding_dim),
    layers.Bidirectional(
        layers.LSTM(max_seqlen, return_sequences=True)),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

absa_model.build(input_shape=(batch_size, max_seqlen))

absa_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-3),
              loss='mae' ,
              metrics=[rmse])


K.clear_session()
history = absa_model.fit(aspect_dataset,
                         epochs=8,
                         steps_per_epoch=300,
                         validation_data=val_dataset_absa,
                         verbose=1,
                         )

mean_score = list(history.history['val_rmse'])
print(mean_score)
loss = np.round(np.mean(mean_score), 2)
print(f'Train_RMSE: {loss}\n')

# Saving the model
def make_dir(dir_name: str):
    """Creates a new directory in the current working directory."""
    save_dir = os.path.join('./', dir_name)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    else:
        print(f'{save_dir}: Already exists!') 
    return save_dir

make_dir('./saved_models')
absa_model.save('./saved_models/sentiment_model.h5')

# Making predictions
predictions = absa_model.predict(test_batched)

print(f'Absa_predictions_shape: {predictions.shape}')

test_data_absa = test_df[['text']].copy()

# Test RMSE:
def _rmse(y_true, y_pred):
    """Computing RMSE metric (without keras backend)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

polarity_test_rmse = _rmse(predictions, polarity_labels_test.values)
print(f'\nTest_RMSE: {polarity_test_rmse}\n')

test_scores = []
for i in range(len(polarity_encoder.classes_)):
    loss = log_loss(polarity_labels_test[polarity_encoder.classes_[i]], predictions[:,i]) 
    test_scores.append(loss)
print(f'Mean Test_loss_polarity: {np.mean(test_scores)}')

