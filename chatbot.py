import random
import json # used to read the intents file
import pickle # use to read the words and classes
import numpy as np
import nltk # Natural Language Toolkit, a library for working with human language data

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents= json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


model = load_model('REMO_chatbot_model.h5')

#This function takes a sentence as input and cleans it up by tokenizing and lemmatizing the words
def clean_up_sentence(sentence):
    # Tokenize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word and convert to lowercase
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#This function creates a bag of words for the input sentence
def bag_of_words(sentence):
    # Initialize the bag of words with zeros
    bag = [0] * len(words)
    # Tokenize and lemmatize the input sentence
    sentence_words = clean_up_sentence(sentence)
    # Set the corresponding indices to 1 for each word in the sentence
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#This function predicts the class of the input sentence
def predict_class(sentence):
    # Create a bag of words for the input sentence
    p = bag_of_words(sentence)
    # Get the model's prediction
    res = model.predict(np.array([p]))[0]
    # Get the indices of the classes with a threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#This function 
def get_response(intents_list, intents_json):
    # Get the intent with the highest probability
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            return response
    return "Sorry, I didn't understand that."  # fallback in case tag not found


print("Chatbot is running...")

while True:
    # Get user input
    message = input("")
    # Predict the class of the input message
    intents_list = predict_class(message)
    # Get the response based on the predicted class
    response = get_response(intents_list, intents)
    print(response)
