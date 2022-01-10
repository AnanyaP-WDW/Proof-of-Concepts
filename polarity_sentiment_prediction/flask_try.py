from flask import Flask, request, render_template, Response

import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
import nltk
from source import scraper
from source import main


from nltk.corpus import stopwords
from matplotlib import pyplot as plt
#from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    """
    for rendering results
    """
    query = request.form["topic name"]
    test_data = main.get_data(query)
    render_data = test_data
    #print("====== Scraping complete ======")
    train_vectors, count_vectorizer = main.count_vectorize(main.train_df['text'])

    ## Map the tokens in the train vectors to the test set.
    # i.e.the train and test vectors use the same set of tokens
    test_vectors = count_vectorizer.transform(main.test_df['text'])

    # split data into features and labels for training
    y_aspect = main.train_df['aspect_term'] #aspect_term target

    y_polarity = main.train_df['polarity'] #polarity target

    aspect_model, sentiment_model = main.build_models()
    print("====== Training in progress ======")
    polarity_model = main.train_absa_model(train_vectors, y_polarity, sentiment_model, task='polarity')
    results,wordcloud_1 = main.inverse_predict(test_data, count_vectorizer, polarity_model)
    plt.savefig('/home/ananya/Desktop/python/trial/absa/static/wordcloud.png')

    return render_template('index.html',train_data = main.train_df.head(10),
    prediction_text = render_data.head(10),
    text_2 = results, plot = 'wordcloud', url = '/home/ananya/Desktop/python/trial/absa/static/wordcloud.png')

# def render_plot():
#     return render_template('index.html', plot = 'wordcloud', url ='/home/ananya/Desktop/python/trial/absa/Static/wordcloud.png')

# @app.route('/image',methods = ['GET'])
# def pic():
#     img_url = url_for('static', filename='wordcloud.png')
#     return render_template('index.html',plot = 'wordcloud', url = img_url)

if __name__ == "__main__":
    #test_data = get_data()

    #aspect_model, sentiment_model = build_models()
    #absa_model = train_absa_model(train_vectors, y_aspect, aspect_model)

    #polarity_model = train_absa_model(train_vectors, y_polarity, sentiment_model, task='polarity')
    #print('==============Running sentiment predictions=================')
    #predict_on_test(test_vectors, polarity_model, task='polarity')

    # Run model against scraped twitter data
    #inverse_predict(test_data, count_vectorizer, polarity_model)

    app.run(debug = True)

#test_data = get_data("burger")
#test_data.head(10)


# import yake
# extractor = yake.KeywordExtractor(lan='en',
#                                   n=1,
#                                   top=5
#                                   )
#
# abc = ["Juri said he canreally cook rice then Taiga said thenll teach you SixTONESANN",
# "aiga said his father ate the rice that he cooked then his father said it might be more delicious than his mom s",
# "Rice was to white"]
#
# data = np.array(abc)
# text = data.flatten().tolist()
# for i, c in enumerate(data):
#     #word_to_vec = tokenizer.transform(c)
#     aspects = extractor.extract_keywords(text[i])
#     # bar plots
#     #plt.bar(dict(list(aspects)).keys(), dict(list(aspects)).values())
#     # Create the wordcloud object
#     #wordcloud = WordCloud(width=480, height=480, margin=0).generate(' '.join(list(dict(list(aspects)).keys())))
#     # Display the generated image:
#     #plt.imshow(wordcloud, interpolation='bilinear')
#     #plt.axis("off")
#     #plt.margins(x=0, y=0)
#     #plt.savefig(f'./wcloud_{i}.png')
#     #plt.show()
#     top_aspect_term = list(aspects)[-1][1]
#
#     top_aspect_term
