#importando bibliotecas necessárias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics as sts

df = pd.read_csv('/home/keven/Documents/Anaconda-spider/datasets/archive/diabetes.csv')
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

scores = []
k = []
i=0
w=0
me = 0
aux = 0
aux2 = 0.05
best = 0
score = 0
p = 0
while(i < 500):
    i += 1
        
    aux = best

    forest = RandomForestClassifier(n_estimators=i, n_jobs=-1 ,random_state=(0))
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    
    scores.append(accuracy_score(y_test, y_pred))
    k.append(i)   
    
    p = accuracy_score(y_test, y_pred)
    
    if (p > aux):
        best = p
        w = i
        if (aux < aux2):
            me = p
            aux2 = p
        else:
            me = aux2
    else:
        best = aux
        
                    
    plt.plot(k, scores)
    plt.show()
    
    print('\n\nmedian: ', sts.median(scores),
          '\nmean: ', sts.mean(scores),
          '\nscore: ', p,
          '\nbest: ', best,
          '\nworst: ', me,
          #'\nworst-generation', pg,
          '\nbest-generation(n_estimators): ',w,
          '\ngeneration: ', i)