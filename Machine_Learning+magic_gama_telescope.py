import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()


df["class"] = (df["class"] == "g").astype(int)
df.head()


for label in cols[:-1]:
  plt.hist(df[df["class"]==1][label], color='blue', label='gama', alpha=0.7, density=True)
  plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()


#Train, Validation, test dataset

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
  x = dataframe[dataframe.columns[:-1]].values    #x = dataframe[dataframe.columns[:-1]].values - Essa linha extrai as colunas de características do DataFrame dataframe, indexando dataframe.columns[:-1]. Ela seleciona todas as colunas, exceto a última (supondo que a última coluna contenha a variável alvo). O atributo .values retorna uma matriz NumPy contendo os valores das colunas selecionadas, que são armazenados na variável x.
  y = dataframe[dataframe.columns[-1]].values     #y = dataframe[dataframe.columns[-1]].values - Essa linha extrai a coluna da variável alvo do DataFrame dataframe, indexando dataframe.columns[-1]. Ela seleciona a última coluna. O atributo .values retorna uma matriz NumPy contendo os valores da coluna selecionada, que são armazenados na variável y.

  scaler = StandardScaler()                       #scaler = StandardScaler() - Essa linha cria uma instância da classe StandardScaler da biblioteca scikit-learn. O StandardScaler é uma etapa de pré-processamento comumente usada para padronizar as características de um conjunto de dados, subtraindo a média e escalando para a variância unitária.
  x = scaler.fit_transform(x)                     #x = scaler.fit_transform(x) - Essa linha aplica o método fit_transform do StandardScaler à matriz de características x. Ele calcula a média e o desvio padrão de cada característica em x usando o método fit e, em seguida, transforma x usando os parâmetros calculados. O resultado é armazenado novamente na variável x.

  if oversample:
    ros = RandomOverSampler()                     #ros = RandomOverSampler() - Essa linha cria uma instância da classe RandomOverSampler da biblioteca imbalanced-learn. O RandomOverSampler é usado para oversampling da classe minoritária para lidar com problemas de desbalanceamento de classes no conjunto de dados.
    x, y = ros.fit_resample(x, y)                 #x, y = ros.fit_resample(x, y) - Se oversample for True, essa linha aplica o método fit_resample do RandomOverSampler à matriz de características x e à matriz alvo y. Ele faz oversampling dos dados, criando amostras sintéticas da classe minoritária até que ambas as classes estejam balanceadas. A matriz de características oversampled e a matriz alvo resultante são armazenadas novamente nas variáveis x e y, respectivamente.

  data = np.hstack((x, np.reshape(y, (-1, 1))))   #data = np.hstack((x, np.reshape(y, (-1, 1)))) - Essa linha empilha horizontalmente a matriz de características escalonadas x e a matriz alvo y usando np.hstack(). A matriz alvo y é remodelada usando np.reshape(y, (-1, 1)) para garantir que ela tenha o mesmo número de dimensões que a matriz de características. Os dados empilhados são armazenados na variável data.

  return data, x, y



train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)


len(y_train)
sum(y_train == 1)
sum(y_train == 0)

len(train[train["class"]==1])  #gamma
len(train[train["class"]==0])

train


#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)

print(classification_report(y_test, y_pred))


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

y_pred = nb_model.predict(x_test)
print(classification_report(y_test, y_pred))

# Log Regression

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(x_train, y_train)

y_pred = lg_model.predict(x_test)
print(classification_report(y_test, y_pred))


from sklearn.svm import SVC


svm_model = SVC()
svm_model = svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)
print(classification_report(y_test, y_pred))