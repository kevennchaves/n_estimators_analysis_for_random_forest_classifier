import pandas as pd
from sklearn.model_selection import train_test_split
from fit import Fit


j = 1
r = 1
e = list()
k = list()
melhor = 0
anterior = 0
scores = list()
random_state = 1234

df = pd.read_csv('/home/keven/Documents/PythonScripts/n_estimators analysis/diabetes.csv')
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

calculo = Fit()

if __name__ == '__main__':    
    for i in range(1,50):
        anterior = melhor
        p = calculo.forest(X_train, X_test, y_train, y_test, i, random_state)     
        
        melhor = calculo.melhor(p, anterior)
        pior = calculo.pior(p, r)
        ideal = calculo.melhor_estimador(p, anterior, i)
        geracao = calculo.generation(i)     
        e.append(calculo.error(p))
        k.append(i)
        scores.append(p)
        r = pior  
        
        if ideal == i:
            j = i
        else:
            j = j        
  
    calculo.plots(k, scores, e)
    calculo.__str__(scores, p, melhor, r, j, geracao)    