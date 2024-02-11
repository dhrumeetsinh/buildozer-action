from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
import os
import pandas as pd
from ntscraper import Nitter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def get_sentiment(text):
    return sia.polarity_scores(text)['compound']

def process_tweets(df):
    df['sentiment_score'] = df['text'].apply(get_sentiment)
    df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

class SentimentAnalysisApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.hashtag_input = TextInput(hint_text='Enter hashtag', multiline=False)
        self.layout.add_widget(self.hashtag_input)

        self.tweets_input = TextInput(hint_text='Enter number of tweets', multiline=False, input_filter='int')
        self.layout.add_widget(self.tweets_input)

        self.analyze_button = Button(text='Analyze Sentiment')
        self.analyze_button.bind(on_press=self.analyze_sentiment)
        self.layout.add_widget(self.analyze_button)

        self.result_label = Label(text='')
        self.layout.add_widget(self.result_label)

        # Register the on_key_down event
        Window.bind(on_key_down=self.on_key_down)

        return self.layout

    def on_key_down(self, window, key, *args):
        if key == 27:  # Escape key code
            App.get_running_app().stop()

    def analyze_sentiment(self, instance):
        hashtag = self.hashtag_input.text
        num_tweets = int(self.tweets_input.text)

        scraper = Nitter()
        tweets = scraper.get_tweets(hashtag, mode='hashtag', number=num_tweets)

        final_tweets = []
        for x in tweets['tweets']:
            data = [x['link'], x['text'], x['date'], x['stats']['likes'], x['stats']['comments']]
            final_tweets.append(data)

        dat = pd.DataFrame(final_tweets, columns=['twitter_link', 'text', 'date', 'likes', 'comments'])

        process_tweets(dat)

        all_text = ' '.join(dat['text'])
        self.generate_wordcloud(all_text)

        self.plot_histogram(dat)

        # Display tweets and sentiments in the result_label
        result_text = "\n".join([f"{row['text']} - {row['sentiment']}" for _, row in dat.iterrows()])
        self.result_label.text = result_text

        # Stop the app when analysis is complete
        App.get_running_app().stop()

    def generate_wordcloud(self, text):
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(text)

        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    def plot_histogram(self, df):
        plt.figure(figsize=(8, 6))
        df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

if __name__ == '__main__':
    SentimentAnalysisApp().run()
