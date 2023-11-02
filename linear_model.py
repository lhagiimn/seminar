from sklearn.datasets import fetch_openml
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

df_boston = fetch_openml(name="boston", as_frame=True)['frame']
input_cols = df_boston.columns.tolist()
target = 'MEDV'
input_cols.remove(target)

train_df, test_df = train_test_split(df_boston,
                                     test_size=0.2,
                                     random_state=42,
                                     shuffle=True)
for col in input_cols:
    train_df[col] = train_df[col].astype('float32')
    test_df[col] = test_df[col].astype('float32')

reg = LinearRegression().fit(train_df[input_cols], train_df[target])
pred = reg.predict(test_df[input_cols])

print('MSE:', mean_squared_error(test_df[target], pred))
print('R2 score:', r2_score(test_df[target], pred))

mse_lasso = 100
final_pred = None
for alpha in [0, 0.01, 0.5, 1, 2, 4, 5]:
    reg_lasso = Ridge(alpha=alpha).fit(train_df[input_cols], train_df[target])
    pred_lasso = reg_lasso.predict(test_df[input_cols])
    curr_mse = mean_squared_error(test_df[target], pred_lasso)
    if curr_mse<mse_lasso:
        print(alpha, curr_mse)
        mse_lasso = curr_mse
        final_pred = pred_lasso

print()
print('Lasso MSE:', mean_squared_error(test_df[target], final_pred))
print('Lasso R2 score:', r2_score(test_df[target], final_pred))
# plt.hist(df_boston[target])
# plt.show()

plt.scatter(x=pred, y=test_df[target].values)
plt.show()