import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
# ==========================================
# 1. 讀取資料 (使用清洗過的 CSV)
# ==========================================
DATA_FILE = 'data/processed/cmp_final_dataset.csv'
print(f"正在讀取 {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

df = df.set_index('WAFER_ID')

# 確保資料按照真實加工時間排序，這對 TimeSeriesSplit 絕對關鍵
if 'START_TIMESTAMP' in df.columns:
    df = df.sort_values('START_TIMESTAMP').drop(columns=['START_TIMESTAMP'])
else:
    df = df.sort_index()

# ==========================================
# 2. 切分 X 與 y
# ==========================================
target_col = 'AVG_REMOVAL_RATE'
X = df.drop(columns=[target_col])

# 若讀到舊版 CSV，強制剔除 neighbor_feature
if 'neighbor_feature' in X.columns:
    X = X.drop(columns=['neighbor_feature'])

# 移除高度共線的耗材特徵，只保留 USAGE_OF_MEMBRANE
if 'USAGE_OF_BACKING_FILM' in X.columns:
    X = X.drop(columns=['USAGE_OF_BACKING_FILM'])

y = df[target_col]

# ==========================================
# 2.1 定義分組邏輯 (Split Logic)
# ==========================================
print("-" * 30)
print("正在根據 Chamber/Stage 路徑進行資料分組 (Segmented Modeling)...")

# 定義 High Group 的特徵關鍵字 (Stage A, Chamber 1, 2, 3)
# 只要 Wafer 在這些 Chamber 有數據 (非 NaN)，就歸類為 High Group
high_group_keywords = ['_Ch1_StgA', '_Ch2_StgA', '_Ch3_StgA']
high_group_cols = [c for c in X.columns if any(k in c for k in high_group_keywords)]

# 建立遮罩 (Mask): 檢查這些欄位是否包含有效值
# 注意：preprocessing.py 產出的 CSV 在未經歷 Chamber 時會是 NaN
is_high_group = X[high_group_cols].notna().any(axis=1)

# 分割訓練資料
X_high = X[is_high_group].copy()
y_high = y[is_high_group].copy()

X_low = X[~is_high_group].copy()
y_low = y[~is_high_group].copy()

print(f"High Group (Ch1-3 StgA) 樣本數: {len(X_high)} (MRR 預期較高)")
print(f"Low Group (Ch4-6 StgA)  樣本數: {len(X_low)}  (MRR 預期較低)")

# ==========================================
# 2.2 特徵剪枝 (Feature Pruning)
# ==========================================
# 定義 Low Group 的特徵 (Ch4-6)
low_group_keywords = ['_Ch4_StgA', '_Ch5_StgA', '_Ch6_StgA']
low_group_cols = [c for c in X.columns if any(k in c for k in low_group_keywords)]

# High Group 不需要 Low Group 的特徵 (Ch4-6)
X_high = X_high.drop(columns=X_high.columns.intersection(low_group_cols))

# Low Group 不需要 High Group 的特徵 (Ch1-3)
X_low = X_low.drop(columns=X_low.columns.intersection(high_group_cols))

print(f"High Group (Ch1-3 StgA) 樣本數: {len(X_high)} (MRR 預期較高)")
print(f"   -> 移除無用特徵後維度: {X_high.shape[1]}")
print(f"Low Group (Ch4-6 StgA)  樣本數: {len(X_low)}  (MRR 預期較低)")
print(f"   -> 移除無用特徵後維度: {X_low.shape[1]}")

# 分別填補缺失值 (Imputation)
imputer_high = SimpleImputer(strategy='constant', fill_value=0)
X_high_filled = imputer_high.fit_transform(X_high)

imputer_low = SimpleImputer(strategy='constant', fill_value=0)
X_low_filled = imputer_low.fit_transform(X_low)

# ==========================================
# 3. 分別訓練兩個模型
# ==========================================
param_grid = {
    'n_estimators': [100,300,500,800,1000],
    'learning_rate': [0.03],
    'max_depth': [4, 5],        
    'min_child_weight': [1,3, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0.1],
    'reg_lambda': [1.0]
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
    xgb = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=inner_cv, # <--- 內層 CV 使用 TimeSeriesSplit
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=1
        )

    # 3. 執行巢狀交叉驗證 (外層)
    print(f"[{group_name}] 正在計算巢狀交叉驗證分數 (這可能需要一些時間)...")
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

    # 5. 為了後續特徵挑選 (SelectFromModel)，在整個資料集上重新訓練以找出最終模型
    print(f"[{group_name}] 正在整個資料集上重新訓練以找出最佳參數與最終模型...")
    grid.fit(X_data, y_data)
    print(f"[{group_name}] 在完整資料集上找到的最佳參數: {grid.best_params_}")
    
    return grid.best_estimator_

# 訓練 High Model
model_high = train_model(X_high_filled, y_high, "High_Group")

# 訓練 Low Model
model_low = train_model(X_low_filled, y_low, "Low_Group")

# ==========================================
# 5. 特徵重要性 (分別查看)
# ==========================================
def show_importance(model, feature_names, title, top_n=30):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print(f"\n【{title} 特徵重要度 Top {top_n}】")
    print(feature_importance_df.head(top_n))
    
    # 繪製長條圖視覺化
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(top_n)
    plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color='steelblue')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(f'{title} - Top {top_n} Feature Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

show_importance(model_high, X_high.columns, "High Group Model", top_n=30)
show_importance(model_low, X_low.columns, "Low Group Model", top_n=30)

# ==========================================
# 5.1 自動特徵篩選與重新訓練 (Refinement)
# ==========================================
print("-" * 30)
print("【正在進行特徵篩選與模型優化 (Refinement)】")

def refine_model(base_model, X_df, y, group_name, cumulative_threshold=0.80):
    # 1. 取得特徵重要度並排序
    importances = base_model.feature_importances_
    imp_df = pd.DataFrame({'Feature': X_df.columns, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # 2. 計算累積重要度
    imp_df['Cumulative_Importance'] = imp_df['Importance'].cumsum()
    
    # 3. 找出剛好跨越累積門檻 (預設 80%) 的特徵索引位置
    threshold_idx = imp_df[imp_df['Cumulative_Importance'] >= cumulative_threshold].index.min()
    
    # 防呆機制：萬一所有特徵加總都不到該門檻
    if pd.isna(threshold_idx):
        threshold_idx = len(imp_df) - 1
        
    # 4. 選出累積達到 80% 資訊量的所有特徵 (包含剛好跨過門檻的那一個)
    selected_cols = imp_df.loc[:threshold_idx, 'Feature'].values
    
    print(f"\n[{group_name}] 特徵縮減: {len(X_df.columns)} -> {len(selected_cols)} 個關鍵特徵 (保留前 {cumulative_threshold*100:.0f}% 累積重要度)")
    
    # 重新準備資料 (只保留關鍵特徵)
    X_selected = X_df[selected_cols]
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_selected_filled = imputer.fit_transform(X_selected)
    
    # 重新訓練模型
    print(f"[{group_name}] 正在使用關鍵特徵重新訓練...")
    refined_model = train_model(X_selected_filled, y, f"{group_name}_Refined")
    
    return refined_model, selected_cols, imputer

# 優化 High Group 模型
model_high_opt, cols_high_opt, imputer_high_opt = refine_model(model_high, X_high, y_high, "High_Group")

# 優化 Low Group 模型
model_low_opt, cols_low_opt, imputer_low_opt = refine_model(model_low, X_low, y_low, "Low_Group")

# ==========================================
# 6. 測試集預測與評估 (Validation Set)
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
        # 1. 先移除 Low Group 特徵 (原始邏輯)
        # 2. 再只保留優化後選出的特徵 (cols_high_opt)
        X_test_high = X_test_high[cols_high_opt] 
        X_test_high_filled = imputer_high_opt.transform(X_test_high)
        y_pred_test[is_test_high] = model_high_opt.predict(X_test_high_filled)
        
    if len(X_test_low) > 0:
        # 測試集也要移除對應的欄位
        # 1. 先移除 High Group 特徵
        # 2. 再只保留優化後選出的特徵 (cols_low_opt)
        X_test_low = X_test_low[cols_low_opt]
        X_test_low_filled = imputer_low_opt.transform(X_test_low)
        y_pred_test[~is_test_high] = model_low_opt.predict(X_test_low_filled)

    # 轉回 numpy array 以便計算 metrics
    y_pred_test = y_pred_test.values

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
        
        # 分色畫出不同組別的預測結果
        plt.scatter(y_test[is_test_high], y_pred_test[is_test_high], alpha=0.6, color='red', label='High Group (Ch1-3)')
        plt.scatter(y_test[~is_test_high], y_pred_test[~is_test_high], alpha=0.6, color='blue', label='Low Group (Ch4-6)')
        
        # 畫出對角線 (完美預測線)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel("Actual Removal Rate")
        plt.ylabel("Predicted Removal Rate")
        plt.title("Validation Set: Actual vs Predicted (XGBoost)")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("預測完成。")

except FileNotFoundError:
    print(f"錯誤: 找不到測試集檔案 {TEST_DATA_FILE}，請確認 preprocessing.py 是否已執行並生成該檔案。")
except Exception as e:
    print(f"發生錯誤: {e}")
