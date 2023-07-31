#nota: codul a fost executat pe bucati, in google colab
import pandas as pd
import numpy as np
from tensorflow import keras 
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier # Import Arbore de decizie
from sklearn.model_selection import train_test_split # Import functie partitionare date
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression

col_names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng','oldpeak','slp','caa','thall','output']
# variabila col_names este optionala si o folosim daca nu dorim sa utilizam capul de tabel din fisier, sau daca nu avem in fisier definita coloanelor
df = pd.read_csv("heart.csv", header=None, skiprows = 1, names=col_names) # header=None - nu utilizam capul de tabel

import matplotlib.pyplot as plt
plt.hist(df['chol'])
plt.show()

y=df.output
x=df.drop(columns='output')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training  30% test

#Regresie logistica
x_train, x_test, y_train, y_test = train_test_split(df.drop("output", axis=1), df["output"], test_size=0.3)
# Construim modelul
model = LogisticRegression()
# Antrenam modelul pe setul de antrenament 
model.fit(x_train, y_train)
# testam acuratetea modelului pe setul de testare
acuratete_regresie = model.score(x_test, y_test)

print("Accuracy:", acuratete_regresie)

#Decision trees
clf = DecisionTreeClassifier()#definim modelul
clf = clf.fit(x_train,y_train)#il antrenam
y_pred = clf.predict(x_test)#facem predictia
acuratete_arbore=metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", acuratete_arbore)#masuram acuratetea

!pip install graphviz
!pip install pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng','oldpeak','slp','caa','thall'],class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('heart.png')
Image(graph.create_png())

#Retele neuronale
# Ne preluam datele, x->datelepe baza carora se face output, y-> clasificare binara dorita
x = df.drop("output", axis=1).values
y = df["output"].values

# Impartim modelul in setul de antrenament si test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# construim modelul retelei neuronale
model = Sequential()
model.add(Dense(64, input_dim=x.shape[1], activation='relu'))#introducem un strat cu 64 de unitati(baza straturilor utilizand keras), relu-Rectified Linear Activation Function->permite invatarea dependentelor non-liniare(returneaza daca valorile sunt >0, altfel returneaza 0)
model.add(Dense(1, activation='sigmoid'))#un strat de output cu activare sigmoida, cea mai potrivita pt clasificari binare

# Compilam modelul
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#adam->algoritm de optimizare

# Antrenam modelul
model.fit(x_train, y_train, epochs=200, batch_size=32)

# Evaluam modelul pe setul test
acuratete_retea_neuronala = model.evaluate(x_test, y_test)[1]

print("Accuracy:", acuratete_retea_neuronala)

#Comparatie si determinarea celui mai bun model
a=acuratete_arbore
b=acuratete_retea_neuronala
c=acuratete_regresie

if a>b and a>c:
  print("Cel mai performant algoritm=> Arbore de decizie, cu acuratetea de: ")
else: 
  if b>a and b>c:
    print("Cel mai performant algoritm=> Retea neuronala, cu acuratetea de: ")
  else:
    print("Cel mai performant algoritm=> Regresie logistica, cu acuratetea de: ")
maxim=max(acuratete_arbore, acuratete_regresie, acuratete_retea_neuronala)
print(maxim)