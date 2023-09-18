import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

feedback=pd.read_csv("ACME-HappinessSurvey2020.csv")

#Rename columns 
feedback.columns=['Happy/Unhappy','On Time','As Expected','Ordered Everything','Good Price','Satisfied with courier','Easy App']

#Addign new feature
feedback['Satisfaction'] = feedback.drop(['Happy/Unhappy'], axis=1).mean(axis=1) # average all features 

# preparing the data 
x=feedback.drop(['Happy/Unhappy'], axis=1)
y=feedback['Happy/Unhappy'] 
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=0) # 70% training and 30% test


#Create a svm Classifier
clf = svm.SVC(kernel='linear',C = 1,gamma = 'scale' ,random_state=1) # Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1 Score:",metrics.f1_score(y_test, y_pred))

#Saving Model
pickle.dump(clf, open('model.pkl', 'wb'))