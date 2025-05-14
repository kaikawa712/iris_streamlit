from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os

# 1. Irisデータセットをロード
iris = datasets.load_iris()

# 2. データをDataFrameに変換
features = pd.DataFrame(iris['data'], columns=iris['feature_names'])
target = iris['target']

# 3. モデルを作成
model = RandomForestClassifier()

# 4. モデルに学習させる
model.fit(features, target)

# 5. 保存先フォルダを作成（存在しない場合のみ）
os.makedirs('models', exist_ok=True)

# 6. モデルを保存
with open('models/model_iris.pkl', 'wb') as f:
    pickle.dump(model, f)

print("モデルが保存されました。")
