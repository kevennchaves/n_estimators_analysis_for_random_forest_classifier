from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import statistics as sts

class Fit:
    
    def __init__(self):
        pass                
        
    def melhor(self, prob, anterior):
        if prob > anterior:
            self.best = prob
        else:
            self.best = anterior
            
        return self.best 
    
    def pior(self, prob, r):
        if self.best < prob:
            worst = self.best
        else:
            worst = prob
            
        if worst < r:
            r = worst
        else:
            r = r     
        return r   

    def melhor_estimador(self, prob, anterior, estimators):
        if prob > anterior:
            return estimators
        
    def generation(self, i):
        generation = 0
        generation += i
        return generation
    
    
    def forest(self, X_train, X_test, y_train, y_test, i, random_state, msf):
        forest = RandomForestClassifier(n_estimators=i,
                                        n_jobs=-1,
                                        random_state=random_state)
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        p = accuracy_score(y_test, y_pred)  
        return p
    
    def error(self, accuracy_score):
        return 1 - accuracy_score
    
    def plots(self, x,y,z):
        plt.plot(x, y, label='Evolução do n_estimador')    
        plt.plot(x, z, label='Evolução do erro')
        plt.legend()
        plt.show()
        
        
    def __str__(self, scores, prob, melhor, r, j, generation):
        print('\n\nmedian: ', sts.median(scores),
          		'\nmean: ', sts.mean(scores),
          		'\nscore: ', prob,
          		'\nbest: ', melhor,
          		'\nworst: ', r,
          		'\nbest_estimator: ',j,
          		'\ngeneration: ', generation)