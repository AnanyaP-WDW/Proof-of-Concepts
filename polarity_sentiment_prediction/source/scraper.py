# fetch tweets from twitter
import tweepy
import pandas as pd
import re


def get_twitter_data(search_word):
    """Fetch tweets from twitter for ABSA."""
    # get the data
    consumerKey = "lulAuGwEdpdHK9N2k9rlBffVg"
    consumerSecret = "qUmfkBnxlbaWZ4HKZRaPpS9fTAicU0fgc8EBszC43yboI3hLZp"
    accessToken = "1257406807103778818-ZTnOlTd52FzbO9wSyKwJ9Dhv4CBv9b"
    accessTokenSecret = "llrRS84dOsLY6wQeNYq9TaUvfGgyTi9Jfx9qukLe8J0Mr"

    # authenticate
    authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

    # set access token
    authenticate.set_access_token(accessToken, accessTokenSecret)

    # create the api object
    api = tweepy.API(authenticate, wait_on_rate_limit=True)

    myCount = 10
    #search_word = input("Enter topic name: ")
    public_tweets = api.search(search_word, count=myCount, lang="en")
    unwanted_words = ['@', 'RT', ':', 'https', 'http']
    symbols = ['@', '#']
    single_chars = re.compile(r'\s+[a-zA-Z]\s+') # remove single chars
    data = []
    for tweet in public_tweets:
        text = tweet.text
        textWords = text.split()
        # print(textWords)
        cleaning_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", " ", text).split())
        cleaning_tweet = single_chars.sub('', cleaning_tweet)
        data.append(cleaning_tweet)
    print("===============Tweet Scrapping complete=============")
    data = pd.DataFrame(data)
    return data, str(search_word).split()
