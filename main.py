import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import plotly
from scipy import sparse as sp_sparse
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import plotly.express as px
import plotly.graph_objects as go
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf

def clean(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything except alphabets basically @ and numbers
    text = re.sub("[^a-zA-Z]"," ",text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()
    text = text.split()
    ps = PorterStemmer() ## Apply Stemming
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))] ## Remove Stopwords
    text = ' '.join(text) ## Joining the words in the list to convert back to text string
    return text

def sentimentfunction(tweet, classifier):
    colors = ['lightslategray',] * 5
    colors[0] = 'crimson'
    classifier = joblib.load("labelmodel.pkl")
    tweet = clean(tweet)
    candidate_labels = ['anger', 'anticipation', 'joy', 'trust', 'fear', 'suprise', 'sadness', 'disgust']
    dic = classifier(tweet, candidate_labels)
    fig = go.Figure(data=[go.Bar(
        x=dic['labels'][0:4],
        y=dic['scores'][0:4],
        marker_color=colors # marker color can be a single color value or an iterable
    )])
    fig.update_layout(
        title_text='Tweet Classification'
    )
    return fig

def predictiontDl(tweetlst):

    input_ids = []
    attention_masks = []

    dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    log_dir = 'dbert_model'
    model_save_path = 'dbert_model.h5'

    def create_model(num_classes):
        inps = Input(shape=(260,), dtype='int64')
        masks = Input(shape=(260,), dtype='int64')
        dbert_layer = dbert_model(inps, attention_mask=masks)[0][:, 0, :]
        dense = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dbert_layer)
        dropout = Dropout(0.5)(dense)
        pred = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout)
        model = tf.keras.Model(inputs=[inps, masks], outputs=pred)

        return model

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    trained_model = create_model(3)
    trained_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    trained_model.load_weights(model_save_path)
    dbert_inps = dbert_tokenizer.encode_plus(tweetlst, add_special_tokens=True, max_length=260, pad_to_max_length=True,
                                                 return_attention_mask=True, truncation=True)
    input_ids.append(dbert_inps['input_ids'])
    attention_masks.append(dbert_inps['attention_mask'])

    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)

    model = joblib.load('finalized_model.sav')

    candidate_labels = ['anger', 'anticipation', 'joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust']

    label = []
    label1 = []
    label2 = []
    label3 = []

    classifier = joblib.load("labelmodel.pkl")
    dic = classifier(tweetlst, candidate_labels)
    label.append(dic['labels'][0])
    label1.append(dic['labels'][1])
    label2.append(dic['labels'][2])
    label3.append(dic['labels'][3])

    preds = trained_model.predict([input_ids, attention_masks], batch_size=8)
    pred_labels = preds.argmax(axis=1)
    return pred_labels[0],label[0],label1[0],label2[0],label3[0]


def predictml(tweets):
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.sav')
    tokens = tfidf_vectorizer.transform([tweets])

    #
    model = joblib.load('finalized_model.sav')

    candidate_labels = ['anger', 'anticipation', 'joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust']

    label = []
    label1 = []
    label2 = []
    label3 = []
    prediction = model.predict(tokens)
    classifier = joblib.load("labelmodel.pkl")
    dic = classifier(tweets, candidate_labels)
    label.append(dic['labels'][0])
    label1.append(dic['labels'][1])
    label2.append(dic['labels'][2])
    label3.append(dic['labels'][3])

    # Adjust the predicted values from 0 to 2 to 1 to 3
    prediction = prediction + 1

    return prediction[0],label[0],label1[0],label2[0],label3[0]

def predictmany(model, tweet):
    result = []
    label = []
    label1 = []
    label2 = []
    label3 = []
    if model == "ml":
        if isinstance(tweet, str):
            tweet = clean(tweet)

            # Your prediction logic here
        elif isinstance(tweet, list):
            for t in tweet:
                if isinstance(t, str):
                    cleaned_tweet = clean(t)

                    pred,labelx,labelxx,labelxt,labelxz = predictml(cleaned_tweet)
                    result.append(pred)
                    label.append(labelx)
                    label1.append(labelxx)
                    label2.append(labelxt)
                    label3.append(labelxz)
                else:
                    result.append(None)
        else:
            return result
        return result,label,label1,label2,label3
    else:
        if isinstance(tweet, str):
            tweet = clean(tweet)

            # Your prediction logic here
        elif isinstance(tweet, list):
            for t in tweet:
                if isinstance(t, str):
                    cleaned_tweet = clean(t)

                    pred, labelx, labelxx, labelxt, labelxz = predictiontDl(cleaned_tweet)
                    result.append(pred)
                    label.append(labelx)
                    label1.append(labelxx)
                    label2.append(labelxt)
                    label3.append(labelxz)
                else:
                    result.append(None)
        else:
            return result
        return result, label, label1, label2, label3

def draw_pie(labels, values, title):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title=title)
    st.plotly_chart(fig)


def file_tab():
    st.markdown('## File Upload')
    file_test = st.file_uploader("Upload a CSV file", type="csv")
    model = st.selectbox("Select Model", ["ml", "dl"])
    nrows = st.number_input("Number of rows", min_value=1, max_value=1000, key='nrows')

    if file_test is not None:
        df = pd.read_csv(file_test, encoding='latin')
        df = df.sample(n=int(nrows),replace=True)
        lst = df['tweet'].tolist()
        if st.button('Predict'):
            with st.spinner('Loading...'):
                result,label,label1,label2,label3 = predictmany(model, lst)
                time.sleep(2)  # Simulate some processing time
            df['Face'] = result
            df['Polarity'] = result
            df['Emotion 1'] = label
            df['Emotion 2'] = label1
            df['Emotion 3'] = label2
            df['Emotion 4'] = label3

            df.to_csv('results.csv', index=False)

            # Generate download link
            with open('results.csv', 'rb') as file:
                st.download_button(label='Download Results', data=file, file_name='results.csv', mime='text/csv')


            st.markdown('### Results')
            st.table(df.head(5))

            polarity_counts = df['Polarity'].value_counts()

            # Plot polarity pie chart
            st.markdown('## Polarity Chart')
            draw_pie(polarity_counts.index, polarity_counts.values, 'Polarity Distribution')

            combined_emotions = df[['Emotion 1', 'Emotion 2', 'Emotion 3', 'Emotion 4']].values.flatten()

            # Calculate the count of each emotion
            emotion_counts = pd.Series(combined_emotions).value_counts()

            st.markdown('## Emotions Chart')
            draw_pie(emotion_counts.index, emotion_counts.values, 'Emotions Distribution')


def main():
    st.set_page_config(page_title='Sentiment Dashboard', page_icon=':smiley:')

    tabs = ["Home", "File"]
    current_tab = st.sidebar.radio("Navigation", tabs)

    if current_tab == "Home":
        st.markdown('# Sentiment Dashboard')
        tweet = st.text_area('Tweet Text')

        if st.button('Predict'):
            classifier = joblib.load("labelmodel.pkl")
            result = sentimentfunction(tweet, classifier)
            st.plotly_chart(result)
    elif current_tab == "File":
        file_tab()


if __name__ == '__main__':
    main()





