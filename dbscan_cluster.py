import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


df = pd.read_excel("data/df.xlsx", header=1)
# print(df.columns)
# print(df.iloc[:, 1].unique())
# print(df.iloc[:, 0].unique())

cols = ['Статистик үзүүлэлт', 'Боомтын нэр', 2018, 2019, 2020, 2021, 2022]
# print(df[cols])

label = ['        Гарсан зорчигчид', '        Орсон зорчигчид']
df = df.loc[df['Боомтын нэр']!="Бүгд"]

for lab in label:
    df_sub = df.loc[df['Статистик үзүүлэлт']==lab].reset_index(drop=True)
    df_sub = df_sub.set_index('Боомтын нэр')
    df_sub = df_sub.fillna(df_sub[cols[4:]].mean(axis=0))

    # df_sub[cols[2:]].plot()
    # plt.show()

    scaler = StandardScaler().fit(df_sub[cols[4:]])
    X = scaler.transform(df_sub[cols[4:]])

    min_sill_score = np.inf
    best_eps = None
    for eps in range(5, 1000, 5):
        eps = eps/1000000
        model = DBSCAN(eps=eps, min_samples=3)
        pred = model.fit_predict(X)
        df_sub['label'] = pred

        current_score =  silhouette_score(X, pred)
        if current_score<min_sill_score:
            best_eps = eps
            min_sill_score = current_score

        print(f'Sill score: {np.round(current_score, 3)}',
              f"Best score: {np.round(min_sill_score, 3)}",
              f"Best Eps: {best_eps}")

    model = DBSCAN(eps=best_eps, min_samples=3)
    pred = model.fit_predict(X)
    df_sub['label'] = pred

    colors = {-1:'red', 0:'blue', 1:'yellow'}
    for i, row in df_sub.iterrows():
        plt.scatter(y=row[2022], x=row['label'], label=i, color=colors[row['label']])
    plt.legend()
    plt.show()


