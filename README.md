# n_estimators_analysis_for_better_use_of_random_forest_in_diabetes_dataset
n_estimators is a parameter from machine learning techniques, called Random Forest. It's a of the techniques more used for programmers.
For resolve problem of searching of best number of tree (n_estimators), in analysis is used of types manipulation for become in a result.

# Introduction
## _Troubleshooting_

Many data scientist, programmers and TI Lovers, face much problems with the switch de some parameters finding in ML techniques.
These parameters are extremely important for get consistent results, that is why, is necessary a analysis for facilitate the
optimal choice for these parameters.

## _Examples of parameters_

- min_samples_leaf
- n_estimators
- n_neighbors

## _Real examples_

- KNN example when n_neigborbs is inconsistent ![KNN](https://s3.amazonaws.com/stackabuse/media/k-nearest-neighbors-algorithm-python-scikit-learn-3.png)
- RandomForest example when n_estimators is inconsistent ![RFC](https://www.researchgate.net/publication/329215185/figure/fig4/AS:697573887180804@1543325983906/Outcome-of-random-forest-regressor-with-varied-parameter-n-estimators.png)

At certain numbers of n_estimators or n neighbors, the model is biased, making the process impossible.
- In KNN: n_neighbors > 5 and < 18
- In RandomForest n_estimators(tree) < 5 

# Objective

Exit the default context where some parameters are fixed. Become them in iterations, to find out best fit.

# Development

## _Language_

For this project was used linguage Python, using IDE Spyder.

## _Dataset_

The dataset choice was diabetes.csv, a basic dataset, even though, much important for analysis. 

## _Machine Learning Technique_

The ML technique used is called Random Forest, that do division of prediction across decision tree, meantime, with it's random.

## _Doing_

### Libraries

Were useds

```sh
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics as sts
```
### Variables
```sh
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
```

### Using sklearn.model_selection

To do download of dataset 'diabets.csv', to do a manipulation even of library pandas.
Firstly to begin, was separate the dataset between, data and target how all algorithms of machine learning.
Configuring the settings:

```sh
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
```

```sh
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
```

### Assign variables to RandomForestClassifier in cycle while

Naturally, when we used random forest, we assign a variable to the model. Example:

```sh
forest = RandomForestClassifier(n_estimators=200, n_jobs=-1 ,random_state=(0))
```

The numbers of n_estimators is fixed, to doing the model to be always in same proportion.
For this answer, it's necessary use a iterator for get the best number os n_estimators (Our objective)
Even, we have:

```sh
while(i < 500):
    i += 1
        
    aux = best

    forest = RandomForestClassifier(n_estimators=i, n_jobs=-1 ,random_state=(0))
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
```

It's possible to improve this loop, add the number of cycle for while accuracy in this moment equals accuracy of random forest.
Example:

>     while(score < 0.77):
>       aux = best
>       forest = RandomForestClassifier(n_estimators=i, n_jobs=-1 ,random_state=(0))
>       forest.fit(X_train, y_train)
>       y_pred = forest.predict(X_test)

But this is a test, even though we will to work only with the first case.

### Get values accuracy in moment

Let's store variables for accuracy this moment.

```sh
 scores.append(accuracy_score(y_test, y_pred))
    k.append(i)    
    
    p = accuracy_score(y_test, y_pred)
```

### Conditions
Conditions for get the best value finding and worst value finding, during the cycle (generation)

```sh
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
```
on what: 
- *best* receive the better accuracy finding
- *me* receive the worst accuracy finding

### End

In the end, ploting the grafics and print the status of the best, worst, just like the cycle (generation) and the best generation.

Example output:

#### median:  0.7922077922077922 
#### mean:  0.7832034632034632 
#### score:  0.7922077922077922 
#### best:  0.8181818181818182 
#### worst:  0.7229437229437229 
#### best-generation(n_estimators):  19 
#### generation:  50

The solution is a best-generation, that was best amount of tree for used in random forest classifier for this exempla, dataset.
But its possible to take advantage of the idea to use in different datasets or situations

# THANK YOU

