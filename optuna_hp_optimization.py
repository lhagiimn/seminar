import optuna
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
Y = iris.target

data = np.concatenate([X, np.expand_dims(Y, axis=1)], axis=1)

trainX, testX = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
trainX, valX = train_test_split(trainX, test_size=0.2, random_state=43, shuffle=True)

def objective(trial):

    tree_depth = trial.suggest_categorical('max_depth', [2, 3, 4, 5])
    cr = trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"])
    model_dt = DecisionTreeClassifier(criterion=cr, max_depth=tree_depth).fit(trainX[:, :-1], trainX[:, -1:])
    pred = model_dt.predict(valX[:, :-1])
    auc = accuracy_score(valX[:, -1:], pred)

    return auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=8)

print(study.best_params)
