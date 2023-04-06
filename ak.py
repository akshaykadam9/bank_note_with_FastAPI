

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

class Akshay:
    def __init__(self, link='https://raw.githubusercontent.com/krishnaik06/FastAPI/main/BankNote_Authentication.csv'):
        self.link = link
        self.df = pd.read_csv(self.link)

    def split(self):
        X = self.df.drop('class', axis=1)
        y = self.df['class']
        return X, y

    def train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def classify(self, X_train, y_train):
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        return classifier

    def pred(self, classifier, X_test):
        y_pred = classifier.predict(X_test)
        return y_pred

    def get_accuracy_score(self, y_test, y_pred):
        score = accuracy_score(y_test, y_pred)
        return score

    def save_model(self, classifier):
        with open('classifier.pkl', 'wb') as f:
            pickle.dump(classifier, f)

if __name__ == '__main__':
    a = Akshay()
    X, y = a.split()
    X_train, X_test, y_train, y_test = a.train_test_split(X, y)
    classifier = a.classify(X_train, y_train)
    y_pred = a.pred(classifier, X_test)
    score = a.get_accuracy_score(y_test, y_pred)
    print(f'Accuracy: {score}')
    a.save_model(classifier)
