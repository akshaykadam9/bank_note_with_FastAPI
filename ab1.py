import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
class akshay:
    def __init__(self,link='https://raw.githubusercontent.com/krishnaik06/FastAPI/main/BankNote_Authentication.csv'):
        self.link=link
    def read_csv(self):
        df=pd.read_csv(self.link)
        return df
    def split(self,df):
        X=df.drop('class',axis=1)
        y=df['class']
        return X,y
    def train_test_split(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test
        #print(_train)
    def classify(self,X_train,y_train):
      classifier=RandomForestClassifier()
      classifier.fit(X_train,y_train)
    def pred(self,X_test):
      y_pred=classifier.predict(X_test)
    def accuracy_score(self):
      score=accuracy_score(y_test,y_pred)
      return score
    def pickel(self):
      pickle_out=open('classifier.pkl','wb')
      pickle.dump(classifier,pickle_out)
      pickle_out.close()
a=akshay()

if __name__ == '__main__':
    df=a.read_csv()
    c,d=a.split(df)
    cl,bl,al,kl=a.train_test_split(c,d)
    y_pred=a.classify(cl,al)
    a.pred(bl)

