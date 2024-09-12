import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap
import os
import seaborn as sns


os.chdir('/home/tyblondr/PycharmProjects/GRAF/zc_combine/data')

df = pd.read_csv('nb201_cached_vkdnw904156120240902_192538.csv')
df = df.set_index('Unnamed: 0')

df_target = pd.read_csv('nb201_val_accs.csv')
df_target = df_target.set_index('net')

df = df.join(df_target, how='left')

target = 'val_accs'
pred_list = []
for p in df.columns:
    if 'tenas' in p:
        pred_list.append(p)
    elif 'fisher' in p:
        pred_list.append(p)
    elif p in ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']:
        pred_list.append(p)

train_df, temp_df = train_test_split(df, test_size=0.89, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

scaler = StandardScaler()
scaler.fit(train_df[pred_list])
train_df[pred_list] = scaler.transform(train_df[pred_list])
valid_df[pred_list] = scaler.transform(valid_df[pred_list])
test_df[pred_list] = scaler.transform(test_df[pred_list])

print(f"Training DataFrame: {train_df.shape[0]}")
print(f"\nValidation DataFrame: {valid_df.shape[0]}")
print(f"\nTest DataFrame: {test_df.shape[0]}")

model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Define the objective as regression with squared error
    n_estimators=1000,             # Number of boosting rounds
    learning_rate=0.05,            # Small learning rate to reduce overfitting
    max_depth=5,                   # Maximum depth of a tree to avoid overfitting
    subsample=0.8,                 # Subsample ratio of the training instances
    colsample_bytree=1,          # Subsample ratio of columns when constructing each tree
    gamma=0.001,                       # Minimum loss reduction required to make a further partition
    reg_alpha=1.0,                 # L1 regularization term on weights
    reg_lambda=1.0,                 # L2 regularization term on weights
)
model.fit(
    train_df[pred_list], train_df[target],
    eval_set=[(train_df[pred_list], train_df[target]), (valid_df[pred_list], valid_df[target])],
    verbose=True
)

test_df[target + '_prediction'] = model.predict(test_df[pred_list])
valid_df[target + '_prediction'] = model.predict(valid_df[pred_list])

corr_matrix = valid_df[[target, target+'_prediction','params']].corr(method='kendall')
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()