import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # 神經網路必備！
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
# ==========================================
# 1. 讀取資料 (使用清洗過的 CSV)
# ==========================================
DATA_FILE = 'data/processed/cmp_final_dataset.csv'
df = pd.read_csv(DATA_FILE)

df = df.set_index('WAFER_ID')

# 確保資料按照 WAFER_ID (時間順序) 排序，這對 TimeSeriesSplit 非常重要
df = df.sort_index()

# 2. 分割 X 和 y
X = df.drop(columns=['AVG_REMOVAL_RATE'])

# 若讀到舊版 CSV，強制剔除 neighbor_feature
if 'neighbor_feature' in X.columns:
    X = X.drop(columns=['neighbor_feature'])

y = df['AVG_REMOVAL_RATE']
print(f"原始特徵數量: {X.shape[1]}")

# --- 新的離群值處理邏輯 (Z-Score Clipping) ---
# 說明：這個方法會找出超過「平均值 + 4倍標準差」的離群值，並將其壓到這個上限。
print("正在對 duration 欄位進行 Z-score=4 的截斷處理 (Clipping)...")
duration_cols = [col for col in X.columns if 'duration' in col]
duration_outlier_thresholds = {} # 儲存計算出的門檻，供測試集使用
for col in duration_cols:
    # 1. 使用 Z-Score=4 來「定義」離群值的門檻
    limit = X[col].mean() + 4 * X[col].std()
    duration_outlier_thresholds[col] = limit
    # 2. 直接使用 clip 函數進行截斷
    X[col] = X[col].clip(upper=limit)
    
imputer=SimpleImputer(strategy='constant',fill_value=0)
X_filled=imputer.fit_transform(X)

# ==========================================
# 3. 設定 5-Fold Cross Validation 與 Grid Search
# ==========================================
print("-" * 30)
print("正在進行 Neural Network Grid Search 參數最佳化 (這可能需要一點時間)...")

# 使用 Pipeline 確保在每一折中正確進行標準化 (避免資料洩漏)
pipeline = Pipeline([
    ('scaler', StandardScaler()), # 改回 StandardScaler
    ('mlp', MLPRegressor(max_iter=3000, early_stopping=True, random_state=42, n_iter_no_change=20))
])

# 設定要測試的參數組合
# 注意：MLPRegressor 不支援 Dropout，主要透過 alpha (L2 Regularization) 來防止過擬合
param_grid = {
    'mlp__hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
    'mlp__alpha': [0.1], # L2 Regularization (Regularization term)
    'mlp__activation': ['relu'],
    'mlp__learning_rate_init': [0.0005, 0.001, 0.01],
    'mlp__batch_size': [16,32,64], # 加入 batch_size 進行網格搜索
    'mlp__solver': ['adam'],
}

# 1. 定義內外層的 CV Splitter (導入 Sliding Window 概念)
n_splits_outer = 5
n_splits_inner = 3

# 為了避免早期折數資料太少導致 MSE 爆掉，我們改用「滑動窗口 (Sliding Window)」
# 動態計算每折的大小，並限制 max_train_size，讓每次訓練的模型都看一樣多的歷史資料
fold_size = len(X_filled) // (n_splits_outer + 1)
max_train_size_outer = fold_size * 2 # 每次最多只拿 2 個 Fold 的歷史資料來訓練

outer_cv = TimeSeriesSplit(n_splits=n_splits_outer, max_train_size=max_train_size_outer)
inner_cv = TimeSeriesSplit(n_splits=n_splits_inner)

# 2. 準備模型與 Grid Search (內層)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=inner_cv, # <--- 內層 CV 使用 TimeSeriesSplit
    scoring='neg_mean_squared_error',
    n_jobs=1, # Windows fix
    verbose=1
)

# 3. 執行巢狀交叉驗證 (外層)
print("正在計算巢狀交叉驗證分數 (神經網路運算非常耗時，可能需要數十分鐘甚至更久，請耐心等候)...")
nested_scores = cross_val_score(grid_search, X_filled, y, cv=outer_cv, scoring='neg_mean_squared_error')

print("-" * 20)
print(f"✅ 巢狀交叉驗證 (Nested CV) 各折 MSE 結果:")
for i, score in enumerate(nested_scores):
    print(f"   Fold {i+1}: {-score:.2f}")

nested_mse = -np.mean(nested_scores)
print(f"   --------------------")
print(f"   平均 MSE: {nested_mse:.2f}")
print("-" * 20)

# 4. 在整個資料集上重新訓練以找出最佳參數與最終模型
print("正在整個資料集上重新訓練以找出最佳參數與最終模型...")
grid_search.fit(X_filled, y)

best_model = grid_search.best_estimator_

# 取得最佳模型的預測結果 (全資料集)
y_pred = best_model.predict(X_filled)

# ==========================================
# 4. 評估結果
# ==========================================
print("-" * 30)
print("【最終模型 最佳參數】")
print(f"最佳參數組合: {grid_search.best_params_}")
print("-" * 30)

# ==========================================
# 5. 畫圖：Loss Curve (訓練過程)
# ==========================================
# 看看模型是怎麼變聰明的
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 1)
# 從 Pipeline 中取出 MLP 模型
mlp_model = best_model.named_steps['mlp']
plt.plot(mlp_model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.grid(True)
ax1.plot(mlp_model.loss_curve_, label='Training Loss', color='blue')
ax1.set_title("Training Loss & Validation Curve")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss (MSE)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# 若有啟用 Early Stopping，繪製內部的 Validation Score 曲線
if hasattr(mlp_model, 'validation_scores_'):
    ax2 = ax1.twinx()  # 建立共用 X 軸的第二個 Y 軸
    ax2.plot(mlp_model.validation_scores_, label='Validation Score (R2)', color='orange')
    ax2.set_ylabel("Validation Score (R2)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # 合併兩個軸的圖例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
else:
    ax1.legend(loc='upper right')

# ==========================================
# 6. 畫圖：預測值 vs 真實值
# ==========================================
plt.subplot(1, 2, 2)
plt.scatter(y, y_pred, alpha=0.5, color='purple')
# 畫一條對角線，預測越接近這條線越準
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Removal Rate")
plt.ylabel("Predicted Removal Rate")
plt.title("Actual vs Predicted (Whole Dataset)")
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 7. 測試集預測與評估 (Validation Set)
# ==========================================
print("-" * 30)
print("【測試集預測與評估】")
TEST_DATA_FILE = 'data/processed/cmp_final_validation_dataset.csv'

try:
    print(f"正在讀取測試集 {TEST_DATA_FILE}...")
    test_df = pd.read_csv(TEST_DATA_FILE)
    
    # 設定 Index
    if 'WAFER_ID' in test_df.columns:
        test_df = test_df.set_index('WAFER_ID')
    
    # 分割 X_test 和 y_test
    if 'AVG_REMOVAL_RATE' in test_df.columns:
        X_test = test_df.drop(columns=['AVG_REMOVAL_RATE'])
        y_test = test_df['AVG_REMOVAL_RATE']
    else:
        X_test = test_df
        y_test = None
        print("警告: 測試集中沒有 'AVG_REMOVAL_RATE' 欄位。")

    # 確保欄位一致性 (處理可能缺少的欄位)
    # 找出訓練集有但測試集沒有的欄位，補 0
    missing_cols = set(X.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    
    # 確保欄位順序一致，並加上 .copy() 避免 SettingWithCopyWarning 導致後續截斷失敗
    X_test = X_test[X.columns].copy()

    # 針對 duration 相關欄位做截斷 (Truncation) 處理
    # 使用訓練集計算出的 Z-score=4 門檻。
    for col, threshold in duration_outlier_thresholds.items():
        if col in X_test.columns:
            X_test[col] = X_test[col].clip(upper=threshold)

    # 填補缺失值 (使用訓練集的 imputer)
    print("正在填補測試集缺失值...")
    X_test_filled = imputer.transform(X_test)

    # 預測
    print("正在進行預測...")
    y_pred_test = best_model.predict(X_test_filled)

    # 評估
    if y_test is not None:
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        print(f"Validation MSE: {mse_test:.2f}")
        print(f"Validation MAPE: {mape_test:.2f}%")
        print(f"Validation R2: {r2_test:.4f}")
        
        # 畫出預測 vs 真實值圖
        plt.figure(figsize=(8, 6))
        
        plt.scatter(y_test, y_pred_test, alpha=0.5, color='green', label='Validation Data')
        # 畫出對角線 (完美預測線)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel("Actual Removal Rate")
        plt.ylabel("Predicted Removal Rate")
        plt.title("Validation Set: Actual vs Predicted (Neural Network)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # ==========================================
        # 8. 特徵重要性分析 (Permutation Importance)
        # ==========================================
        from sklearn.inspection import permutation_importance
        print("-" * 30)
        print("【特徵重要性分析 (Permutation Importance)】")
        print("正在計算特徵重要性 (這可能需要幾分鐘)...")
        
        # 使用驗證集進行計算
        result = permutation_importance(best_model, X_test_filled, y_test, n_repeats=10, random_state=42, n_jobs=1)
        sorted_idx = result.importances_mean.argsort()[::-1]
        
        print("Top 20 Important Features:")
        for i in sorted_idx[:20]:
            print(f"{X.columns[i]}: {result.importances_mean[i]:.4f}")

    else:
        print("預測完成。")

except FileNotFoundError:
    print(f"錯誤: 找不到測試集檔案 {TEST_DATA_FILE}，請確認 preprocessing.py 是否已執行並生成該檔案。")
except Exception as e:
    print(f"發生錯誤: {e}")
