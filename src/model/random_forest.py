import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# 1. 讀取處理好的 CSV
DATA_FILE = 'data/processed/cmp_final_dataset.csv'
print(f"正在讀取 {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
df = df.set_index('WAFER_ID')


if 'START_TIMESTAMP' in df.columns:
    df = df.sort_values('START_TIMESTAMP').drop(columns=['START_TIMESTAMP'])
else:
    df = df.sort_index()

# 2. 分割 X 和 y
X = df.drop(columns=['AVG_REMOVAL_RATE'])


y = df['AVG_REMOVAL_RATE']

# ==========================================
# 2.1 定義分組邏輯 (Split Logic)
# ==========================================
print("-" * 30)
print("正在根據 Chamber/Stage 路徑進行資料分組 (Segmented Modeling)...")

high_group_keywords = ['_Ch1_StgA', '_Ch2_StgA', '_Ch3_StgA']
high_group_cols = [c for c in X.columns if any(k in c for k in high_group_keywords)]


is_high_group = X[high_group_cols].notna().any(axis=1)

X_high = X[is_high_group].copy()
y_high = y[is_high_group].copy()

X_low = X[~is_high_group].copy()
y_low = y[~is_high_group].copy()

# ==========================================
# 2.2 特徵剪枝 (Feature Pruning)
# ==========================================
# 定義 Low Group 的特徵 (Ch4-6)
low_group_keywords = ['_Ch4_StgA', '_Ch5_StgA', '_Ch6_StgA']
low_group_cols = [c for c in X.columns if any(k in c for k in low_group_keywords)]


X_high = X_high.drop(columns=X_high.columns.intersection(low_group_cols))

X_low = X_low.drop(columns=X_low.columns.intersection(high_group_cols))

print(f"High Group (Ch1-3 StgA) 樣本數: {len(X_high)} (MRR 預期較高)")
print(f"   -> 移除無用特徵後維度: {X_high.shape[1]}")
print(f"Low Group (Ch4-6 StgA)  樣本數: {len(X_low)}  (MRR 預期較低)")
print(f"   -> 移除無用特徵後維度: {X_low.shape[1]}")


imputer_high = SimpleImputer(strategy='constant', fill_value=0)
X_high_filled = imputer_high.fit_transform(X_high)

imputer_low = SimpleImputer(strategy='constant', fill_value=0)
X_low_filled = imputer_low.fit_transform(X_low)

# ==========================================
# 3. 分別訓練兩個模型
# ==========================================

param_grid = {
    'n_estimators': [100,200,300,500,600,700],
    'max_depth': [10, 20, 30,40,50],
    'min_samples_leaf': [1,2,3]
}

def train_model(X_data, y_data, group_name):
    print(f"\n正在為 {group_name} 模型執行巢狀時間序列交叉驗證 (Nested Time-Series CV)...")
    
    # 1. 定義內外層的 CV Splitter (導入 Sliding Window 概念)
    n_splits_outer = 5
    n_splits_inner = 3
    
    # 為了避免早期折數資料太少導致 MSE 爆掉，我們改用「滑動窗口 (Sliding Window)」
    # 動態計算每折的大小，並限制 max_train_size，讓每次訓練的模型都看一樣多的歷史資料
    fold_size = len(X_data) // (n_splits_outer + 1)
    max_train_size_outer = fold_size * 2 # 每次最多只拿 2 個 Fold 的歷史資料來訓練

    outer_cv = TimeSeriesSplit(n_splits=n_splits_outer, max_train_size=max_train_size_outer)
    inner_cv = TimeSeriesSplit(n_splits=n_splits_inner)

    # 2. 準備模型與 Grid Search (內層)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=inner_cv, # <--- 內層 CV 使用 TimeSeriesSplit
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=1
    )

    # 3. 執行巢狀交叉驗證 (外層)
    print(f"[{group_name}] 正在計算巢狀交叉驗證分數 (Random Forest 運算需要一點時間)...")
    nested_scores = cross_val_score(grid, X=X_data, y=y_data, cv=outer_cv, scoring='neg_mean_squared_error')
    
    # 4. 報告最嚴謹的效能評估結果
    print("-" * 20)
    print(f"✅ [{group_name}] 巢狀交叉驗證 (Nested CV) 各折 MSE 結果:")
    for i, score in enumerate(nested_scores):
        print(f"   Fold {i+1}: {-score:.2f}")
    
    nested_mse = -np.mean(nested_scores)
    print(f"   --------------------")
    print(f"   平均 MSE: {nested_mse:.2f}")
    print("-" * 20)

    # 5. 在整個資料集上重新訓練以找出最佳參數與最終模型
    print(f"[{group_name}] 正在整個資料集上重新訓練以找出最佳參數與最終模型...")
    grid.fit(X_data, y_data)

    print(f"[{group_name}] 最佳參數: {grid.best_params_}")
    return grid.best_estimator_

model_high = train_model(X_high_filled, y_high, "High_Group")
model_low = train_model(X_low_filled, y_low, "Low_Group")

# ==========================================
# 5. 特徵重要性 (分別查看)
# ==========================================
def show_importance(model, feature_names, title):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print(f"\n【{title} 特徵重要度 Top 10】")
    print(feature_importance_df.sort_values(by='Importance', ascending=False).head(10))

show_importance(model_high, X_high.columns, "High Group Model")
show_importance(model_low, X_low.columns, "Low Group Model")

# ==========================================
# 11. 測試集預測與評估 (Validation Set)
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
    
    # 確保欄位順序一致
    X_test = X_test[X.columns]

    # -------------------------------------------------------
    # 分組預測邏輯
    # -------------------------------------------------------
    print("正在進行分組預測...")
    
    # 1. 判斷測試集分組
    is_test_high = X_test[high_group_cols].notna().any(axis=1)
    
    # 2. 分割測試集
    X_test_high = X_test[is_test_high].copy()
    X_test_low = X_test[~is_test_high].copy()
    
    # 3. 填補缺失值 (使用對應組別的 imputer) & 預測
    y_pred_test = pd.Series(index=X_test.index, dtype=float)
    
    if len(X_test_high) > 0:
        # 測試集也要移除對應的欄位
        X_test_high = X_test_high.drop(columns=X_test_high.columns.intersection(low_group_cols))
        X_test_high_filled = imputer_high.transform(X_test_high)
        y_pred_test[is_test_high] = model_high.predict(X_test_high_filled)
        
    if len(X_test_low) > 0:
        # 測試集也要移除對應的欄位
        X_test_low = X_test_low.drop(columns=X_test_low.columns.intersection(high_group_cols))
        X_test_low_filled = imputer_low.transform(X_test_low)
        y_pred_test[~is_test_high] = model_low.predict(X_test_low_filled)

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
        
        plt.scatter(y_test[is_test_high], y_pred_test[is_test_high], alpha=0.6, color='red', label='High Group (Ch1-3)')
        plt.scatter(y_test[~is_test_high], y_pred_test[~is_test_high], alpha=0.6, color='blue', label='Low Group (Ch4-6)')
        
        # 畫出對角線 (完美預測線)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel("Actual Removal Rate")
        plt.ylabel("Predicted Removal Rate")
        plt.title("Validation Set: Actual vs Predicted (Random Forest)")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("預測完成。")

except FileNotFoundError:
    print(f"錯誤: 找不到測試集檔案 {TEST_DATA_FILE}，請確認 preprocessing.py 是否已執行並生成該檔案。")
except Exception as e:
    print(f"發生錯誤: {e}")
