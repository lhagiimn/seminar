
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
import numpy as np


def fit_lgbm(Nfolds, train_df,
             train_cols, cat_feats,
             TARGET, params: dict = None):
    oof_pred_lgb = np.zeros(train_df.shape[0], dtype=np.float32)
    imp = np.zeros(len(train_cols))
    scores = []

    for fold in range(Nfolds):
        print("*" * 10, f'Fold-{fold + 1}', "*" * 10)
        train_set = train_df.loc[train_df['fold'] != fold]
        val_set = train_df.loc[train_df['fold'] == fold]

        train_data = lgb.Dataset(data=train_set[train_cols],
                                 label=train_set[TARGET],
                                 categorical_feature=cat_feats,
                                 free_raw_data=False)

        valid_data = lgb.Dataset(data=val_set[train_cols],
                                 label=val_set[TARGET],
                                 categorical_feature=cat_feats,
                                 free_raw_data=False)

        model = lgb.train(params, train_data, valid_sets=[train_data, valid_data],
                          num_boost_round=2500)

        oof_pred_lgb[val_set.index] = model.predict(val_set[train_cols])
        score_lgb = (roc_auc_score(val_set[TARGET], oof_pred_lgb[val_set.index]))
        scores.append(score_lgb)
        print(f'AUC for Fold-{fold + 1}:', np.round(score_lgb, 3))

        imp += model.feature_importance(importance_type="gain") / Nfolds

    score_lgb = (roc_auc_score(train_df[TARGET], oof_pred_lgb))
    print(f'OOF AUC:', np.round(score_lgb, 3))
    print(f'Average AUC:', f'{np.round(np.mean(scores), 3)}+/-{np.round(np.std(scores), 3)}')

    return oof_pred_lgb, imp

df_bank = pd.read_csv("data/bank.csv", sep=";")

encoders = {}
cat_cols = []
num_cols = []
target = "y"
for col in df_bank.columns:
    if df_bank[col].dtype=='object':
        enc = LabelEncoder().fit(df_bank[col])
        encoders[col] = enc
        df_bank[col] = enc.transform(df_bank[col])
        if col!="y":
            cat_cols.append(col)
    else:
        num_cols.append(col)

print('Number of continues columns:', len(num_cols))

X = PCA(n_components=2).fit_transform(df_bank[num_cols].values)

# plt.scatter(x=X[:, 0], y=X[:, 1])
# plt.show()

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
score = lof.fit_predict(df_bank[num_cols])

df_bank['outlier_score'] = lof.negative_outlier_factor_
df_bank = df_bank.sort_values(by=['outlier_score'])
print(df_bank[num_cols+['outlier_score']])

# plt.scatter(x=X[:, 1], y = df_bank['outlier_score'].values)
# plt.show()
df_bank['outlier'] = score
df_bank_outlier_removed = df_bank.loc[df_bank['outlier']!=-1].reset_index(drop=True)
df_bank = df_bank.reset_index(drop=True)
print(df_bank.shape)
print(df_bank_outlier_removed.shape)


#Modelling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

df_bank['fold'] = 0
for i, (train_idx, test_idx) in enumerate(kf.split(df_bank.index)):
    df_bank.loc[test_idx, 'fold'] = i


params = {'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'random_state': 42,
          'n_estimators': 3000,
          'reg_alpha': 1.029841665079736703,
          'reg_lambda': 3.8446449556043025,
          'subsample': 0.65,
          'colsample_bytree': 0.40,
          'learning_rate': 0.05,
          'min_child_samples': 43,
          'num_leaves': 262}

oof_pred_lgb, imp = fit_lgbm(Nfolds=5, train_df=df_bank,
                             train_cols=num_cols+cat_cols,
                             cat_feats=cat_cols, TARGET=target,
                             params = params)

feature_imp = pd.DataFrame(
            {
                "Value": imp/5,
                "Feature": num_cols+cat_cols,
            })

feature_imp=feature_imp.sort_values(by='Value', ascending=False)

print(feature_imp)

plt.scatter(x=feature_imp['Feature'], y=feature_imp['Value'])
plt.show()






