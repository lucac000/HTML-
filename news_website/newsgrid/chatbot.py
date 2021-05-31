import random
import json
import pickle
import numpy as np 

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import random
import json
import pickle
import numpy as np 
from flask import redirect, url_for

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

class Chat: 

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

        self.intents = json.loads(open("intents.json").read())

        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ["!","?",".",","]

        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent["tag"]))
                if intent["tag"] not in self.classes:
                    self.classes.append(intent["tag"])

        self.words = [self.lemmatizer.lemmatize(word) for word in self.words if word not in self.ignore_letters]
        self.words = sorted(set(self.words))

        self.classes = sorted(set(self.classes))

        pickle.dump(self.words, open("words.pkl", "wb"))
        pickle.dump(self.classes, open("classes.pkl", "wb"))

        training = []
        output_empty = [0] * len(self.classes)

        for document in self.documents:
            self.bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                self.bag.append(1) if word in word_patterns else self.bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([self.bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation="softmax"))

        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        self.model.save("chatbotmodel.h5", hist)

        self.lemmatizer = WordNetLemmatizer()

        self.intents = json.loads(open("intents.json").read())

        self.words = pickle.load(open("words.pkl", "rb"))
        self.classes = pickle.load(open("classes.pkl", "rb"))
        self.model = load_model("chatbotmodel.h5")


    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        self.bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    self.bag[i] = 1
        return np.array(self.bag)  

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
        return result

    print("GO! StockBot is running")

    def get_answer(self, msg):
        
        ints = self.predict_class(msg)
        res = self.get_response(ints, self.intents)
        return res
