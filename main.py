import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('data/iris.csv')
print(df.columns)

le = LabelEncoder().fit(df['variety'])
df['label'] = le.transform(df['variety'])
#df = df.loc[df['label']<2].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

input_cols = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
for i, (train_idx, test_idx) in enumerate(kf.split(df)):

    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    neigh = KNeighborsClassifier(n_neighbors=15)

    neigh.fit(train_df[input_cols].values, train_df['label'].values)
    pred = neigh.predict(test_df[input_cols])

    clf = DecisionTreeClassifier()
    clf = clf.fit(train_df[input_cols].values, train_df['label'].values)
    pred_dt = clf.predict(test_df[input_cols])

    rf = RandomForestClassifier(n_estimators=500, criterion='entropy')
    rf = rf.fit(train_df[input_cols].values, train_df['label'].values)
    pred_rf = rf.predict(test_df[input_cols])

    print()
    print(f"KNN Fold-{i}:", accuracy_score(test_df['label'], pred))
    print(f"DT Fold-{i}:", accuracy_score(test_df['label'], pred_dt))
    print(f"RF Fold-{i}:", accuracy_score(test_df['label'], pred_rf))

exit()
colors = ['red', 'green', 'blue']
for i, var in enumerate(df['variety'].unique()):
    temp = df.loc[df['variety']==var]
    plt.scatter(x=temp['sepal.length'], y=temp['sepal.width'], color=colors[i])

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('IRIS DATASET')
plt.show()

#print(df.describe())
