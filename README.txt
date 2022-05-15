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

- KNN example when n_neigborbs is inconsistent ![Here](https://drive.google.com/file/d/1sb5fWK0ejaKOO5QMTyO2KNTSutHNe9sN/view?usp=sharing)
- RandomForest example when n_estimators is inconsistent ![Here](https://drive.google.com/file/d/1rEJEH3SLibZcczoNzI2_lpjcwMMG5AoW/view?usp=sharing)

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

The numbers of n_estimators is fix, to doing the model to be always in same proportion.


To do download of dataset 'diabets.csv', to do a manipulation even of library pandas.
Firstly to begin, was separate the dataset between, data and target how all algorithms of machine learning.
Configuring the settings:
