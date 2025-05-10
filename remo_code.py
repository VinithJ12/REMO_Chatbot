import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# This reduces words from their base form to their root form
lemmatizer= WordNetLemmatizer()

# This loads and preprocesses the intents file
intents= json.loads(open('intents.json').read())

words=[]
classes=[]
documents=[]
ignore_words=['?','!','.',',']

#This splits the sentences into words
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList= nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


#This cleans up and standardizes the words
words= [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words= sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training= []
output_empty= [0]*len(classes)

#This code creates a vecotr of 0s and 1s for each word in the sentence
for doc in documents:
    bag= []
    pattern_words= doc[0]
    pattern_words= [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

 #Encodes the tag into a one-hot vector.
    output_row= list(output_empty)
    output_row[classes.index(doc[1])]= 1

    training.append([bag, output_row])

random.shuffle(training)
training= np.array(training)
train_x= training[:, :len(w)]
train_y= training[:, len(w):]

#Neural Network Model

model= tf.keras.Sequential()
#Dense layer is a fully connected layer (which means that every neuron in the layer is connected to every neuron in the previous layer)
#This has a fully connected layer with 128 neurons and a dropout layer to prevent overfitting, also the input shape is the length of the training data
#The Relu activation function helps with non-linearity 
#The Dropout layer randomly turns off (sets to 0) 50% of the neurons during training each step.- prevents overfitting
#If we put the dropout layer value as lower than 0.5, it may learn faster and the model retains more info, while if we put it higher than 0.5, it may learn slower and the model retains less info

#Consider using different activation functions like sigmoid, tanh, softmax, etc. for different layers and different values of dropout ( This is called hyperparameter tuning)
model.add(tf.keras.layer.Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # can we use relu or sigmoid or tanh or softmax or any other activation function
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax')) # softmax is used for multi-class classification

#STOCHASTIC GRADIENT DESCENT
#This is an optimization algorithm that is used to minimize the loss function and update the weights of the model

# The hyperparameters are the learning rate, momentum and nesterov:
# learning rate is the step size at each iteration while moving toward a minimum of the loss function ( smaller is slower but safer, larger is faster but riskier)
# momentum is a technique that helps accelerate SGD in the relevant direction and dampens oscillations ( adds memory of previous gradients to the current gradient)
# nesterov is a variant of momentum that helps to improve the convergence speed of the model ( it looks ahead to see where the gradient is going)
sgd= tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss= 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#This trains the model with the training data and validates it with the test data

#The epochs is the number of times the model will go through the entire training data
#The batch size is the number of samples that will be used in each iteration
#The verbose is the level of detail that will be shown during training (0 = silent, 1 = progress bar, 2 = one line per epoch)
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('REMO_chatbot_model.h5', hist)
#This saves the model to a file
print("Model Created")