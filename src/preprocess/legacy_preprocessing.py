import pandas as pd
import glob
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 設定路徑 (請依你的環境修改)
# ==========================================
DATA_PATH = 'data/raw/training'
LABEL_PATH = 'data/raw/labels/CMP-training-removalrate.csv'
OUTPUT_FILE = 'data/processed/cmp_final_dataset.csv'
VAL_PATH = 'data/raw/test'
LABEL_PATH1 = 'data/raw/labels/CMP-test-removalrate.csv'
TEST_OUTPUT_FILE = 'data/processed/cmp_final_test_dataset.csv'

# ==========================================
# 2. 讀取與合併
# ==========================================
all_files = glob.glob(os.path.join(DATA_PATH, '*.csv'))
df_list = []

print(f"正在讀取 {len(all_files)} 個 CSV 檔案...")
for file in all_files:
    df_temp = pd.read_csv(file)
    df_list.append(df_temp)

big_df = pd.concat(df_list, ignore_index=True)
print("原始資料合併完成。")

print(f"正在讀取測試集資料...")
test_files = glob.glob(os.path.join(VAL_PATH, '*.csv'))
test_files = [f for f in test_files if os.path.basename(f) != os.path.basename(LABEL_PATH1)]


test_big_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
print("測試集資料合併完成。")

# ==========================================
#移除高度共線的特徵
# ==========================================
cols_to_drop = [
    'CENTER_AIR_BAG_PRESSURE',
    'RIPPLE_AIR_BAG_PRESSURE',
    'EDGE_AIR_BAG_PRESSURE'
]

big_df = big_df.drop(columns=[col for col in cols_to_drop if col in big_df.columns], errors='ignore')
test_big_df = test_big_df.drop(columns=[col for col in cols_to_drop if col in test_big_df.columns], errors='ignore')


# ==========================================
# 合併計算各階段時間與純研磨期特徵
# ==========================================
def extract_physics_features(df, name="訓練集"):
    print(f"正在為 {name} 計算各階段時間與純研磨期特徵...")
    df = df.sort_values(by=['WAFER_ID', 'CHAMBER', 'STAGE', 'TIMESTAMP']).copy()

    time_diffs = df['TIMESTAMP'].diff().fillna(0)
    boundary_mask = (df['WAFER_ID'] != df['WAFER_ID'].shift(1)) | \
                    (df['CHAMBER'] != df['CHAMBER'].shift(1)) | \
                    (df['STAGE'] != df['STAGE'].shift(1))
    time_diffs[boundary_mask] = 0

    if 'PRESSURIZED_CHAMBER_PRESSURE' in df.columns:
        pressure_active = df['PRESSURIZED_CHAMBER_PRESSURE'] > 0
    else:
        pressure_active = pd.Series(False, index=df.index)

    head_is_rotating = df['HEAD_ROTATION'] > 0.1 if 'HEAD_ROTATION' in df.columns else pd.Series(False, index=df.index)
    wafer_is_rotating = df['WAFER_ROTATION'] > 0.1 if 'WAFER_ROTATION' in df.columns else pd.Series(False, index=df.index)
    stage_is_rotating = df['STAGE_ROTATION'] > 0.1 if 'STAGE_ROTATION' in df.columns else pd.Series(False, index=df.index)

    polishing_rotation_active = wafer_is_rotating & stage_is_rotating
    any_rotation_active = head_is_rotating | wafer_is_rotating | stage_is_rotating
    is_polishing = pressure_active & polishing_rotation_active

    df['polishing_duration'] = time_diffs.where(is_polishing, 0)
    df['soaking_duration'] = time_diffs.where(pressure_active & ~polishing_rotation_active, 0)
    df['idle_duration'] = time_diffs.where(~pressure_active & ~any_rotation_active, 0)
    df['spinning_duration'] = time_diffs.where(~pressure_active & any_rotation_active, 0)
    
    pressure_cols = [col for col in df.columns if 'PRESSURE' in col and not col.startswith('polishing_')]
    for p_col in pressure_cols:
        df[f"polishing_{p_col}"] = df[p_col].where(is_polishing, np.nan)
        
    print(f"{name} 物理特徵處理完成。")
    return df

# ==========================================
# 3. 耗材正規化 (MinMax)
# ==========================================
print("\n正在檢查主要耗材之間的相關性...")
consumable_cols_to_check = [
    'USAGE_OF_BACKING_FILM', 
    'USAGE_OF_MEMBRANE', 
    'USAGE_OF_PRESSURIZED_SHEET'
]

existing_cols_to_check = [col for col in consumable_cols_to_check if col in big_df.columns]

if len(existing_cols_to_check) > 1:
    correlation_matrix = big_df[existing_cols_to_check].corr()
    
    print("耗材相關係數矩陣:")
    print(correlation_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5)
    plt.title('Correlation Matrix of Consumable Usage', fontsize=14)
    plt.show()
    
    print("正在繪製配對圖 (Pair Plot)...")
    sns.pairplot(big_df[existing_cols_to_check].dropna().sample(n=min(2000, len(big_df.dropna()))))
    plt.suptitle('Pair Plot of Consumable Usage (Sampled)', y=1.02, fontsize=14)
    plt.show()


# ==========================================
# 4. 耗材正規化 (MinMax)
# ==========================================
scaler = MinMaxScaler()
consumable_cols = [
    'USAGE_OF_DRESSER',
    'USAGE_OF_POLISHING_TABLE', 'USAGE_OF_DRESSER_TABLE',
    'USAGE_OF_MEMBRANE' 
]

big_df = extract_physics_features(big_df, name="訓練集")
test_big_df = extract_physics_features(test_big_df, name="測試集")


print("正在進行耗材正規化...")
scalers = {}
for col in consumable_cols:
    if col in big_df.columns:
        s = MinMaxScaler()
        big_df[col] = s.fit_transform(big_df[[col]])
        scalers[col] = s

for col in consumable_cols:
    if col in test_big_df.columns and col in scalers:
        test_big_df[col] = scalers[col].transform(test_big_df[[col]])

# ==========================================
# 5. 特徵工程 (Aggregation)
# ==========================================
def nonzero_mean(x):
    filtered = x[x > 0.1] 
    return filtered.mean() if not filtered.empty else np.nan

def nonzero_std(x):
    filtered = x[x > 0.1] 
    return filtered.std() if not filtered.empty else np.nan

def nonzero_median(x):
    filtered = x[x > 0.1] 
    return filtered.median() if not filtered.empty else np.nan

print("進行 Groupby 與特徵萃取")
agg_dict = {
    'USAGE_OF_DRESSER':['max'],
    'USAGE_OF_POLISHING_TABLE':['max'],
    'USAGE_OF_DRESSER_TABLE':['max'],
    'polishing_duration': ['sum'],
    'soaking_duration': ['sum'],
    'idle_duration': ['sum'],
    'spinning_duration': ['sum'],
    'USAGE_OF_MEMBRANE':['max'],
    'PRESSURIZED_CHAMBER_PRESSURE': [nonzero_mean, nonzero_std],
    'MAIN_OUTER_AIR_BAG_PRESSURE': [nonzero_mean, nonzero_std],
    'RETAINER_RING_PRESSURE':[nonzero_mean, nonzero_std],
    'SLURRY_FLOW_LINE_A':['sum'],
    'SLURRY_FLOW_LINE_B':['sum'],
    'SLURRY_FLOW_LINE_C':['sum'],
    'WAFER_ROTATION':[nonzero_mean, nonzero_std],
    'STAGE_ROTATION':[nonzero_median,nonzero_std],
    'HEAD_ROTATION':[nonzero_mean, nonzero_std],
    'DRESSING_WATER_STATUS':['mean'],
}

original_pressure_cols = [
    'PRESSURIZED_CHAMBER_PRESSURE', 'MAIN_OUTER_AIR_BAG_PRESSURE', 
    'RETAINER_RING_PRESSURE'
]
for p_col in original_pressure_cols:
    polishing_col_name = f"polishing_{p_col}"
    if polishing_col_name in big_df.columns:
        agg_dict[polishing_col_name] = [np.nanmean, np.nanstd]

feature_df = big_df.groupby(['WAFER_ID', 'CHAMBER','STAGE']).agg(agg_dict)
test_feature_df = test_big_df.groupby(['WAFER_ID', 'CHAMBER','STAGE']).agg(agg_dict)

# ==========================================
# 6. 展開與欄位命名 (Unstack)
# ==========================================
feature_unstacked = feature_df.unstack(level=['CHAMBER', 'STAGE'])
test_feature_unstacked = test_feature_df.unstack(level=['CHAMBER', 'STAGE'])

def rename_cols(df):
    new_cols = []
    for col in df.columns:
        var_name = col[0]
        stat_name = col[1]
        chamber = str(int(col[2]))
        stage = col[3]
        
        if hasattr(stat_name, '__name__'):
            stat_name = stat_name.__name__
        
        new_cols.append(f"{var_name}_{stat_name}_Ch{chamber}_Stg{stage}")
    df.columns = new_cols
    return df

feature_unstacked = rename_cols(feature_unstacked)
test_feature_unstacked = rename_cols(test_feature_unstacked)

# ==========================================
# 7. 合併目標值 (Y) 與清洗異常值 (Data Cleaning)
# ==========================================
print("正在合併 Y 值...")
y_df = pd.read_csv(LABEL_PATH)
if 'STAGE' in y_df.columns:
    y_df = y_df.drop(columns=['STAGE'])

y_df = y_df.drop_duplicates(subset=['WAFER_ID']).set_index('WAFER_ID')

final_df = feature_unstacked.join(y_df, how='inner')
train_start_time = big_df.groupby('WAFER_ID')['TIMESTAMP'].min().rename('START_TIMESTAMP')
final_df = final_df.join(train_start_time).sort_values('START_TIMESTAMP')

print("正在合併 Test Y 值...")
y_test_df = pd.read_csv(LABEL_PATH1)
if 'STAGE' in y_test_df.columns:
    y_test_df = y_test_df.drop(columns=['STAGE'])
y_test_df = y_test_df.drop_duplicates(subset=['WAFER_ID']).set_index('WAFER_ID')
final_test_df = test_feature_unstacked.join(y_test_df, how='inner')

test_start_time = test_big_df.groupby('WAFER_ID')['TIMESTAMP'].min().rename('START_TIMESTAMP')
final_test_df = final_test_df.join(test_start_time).sort_values('START_TIMESTAMP')

# ---------------------------------------------------------
# 🔥 過濾異常值 (Data Cleaning)
# ---------------------------------------------------------
print(f"清洗前資料筆數: {len(final_df)}")

UPPER_LIMIT = 300
LOWER_LIMIT = 10 

clean_mask = (final_df['AVG_REMOVAL_RATE'] < UPPER_LIMIT) & \
             (final_df['AVG_REMOVAL_RATE'] > LOWER_LIMIT)


original_count = len(final_df)
final_df = final_df[clean_mask]
cleaned_count = len(final_df)

print(f"清洗後資料筆數: {cleaned_count}")
print(f"🔪 成功踢除了 {original_count - cleaned_count} 筆異常資料")
# ---------------------------------------------------------

# 【關鍵步驟】將 WAFER_ID 從 Index 變成一個欄位
# 這樣 to_csv(index=False) 才不會把它弄不見！
final_df.reset_index(inplace=True)

print(f"最終資料形狀: {final_df.shape}")
print(f"正在儲存為 CSV: {OUTPUT_FILE} ...")

final_df.to_csv(OUTPUT_FILE, index=False)

final_test_df.reset_index(inplace=True)
print(f"正在儲存測試集 CSV: {TEST_OUTPUT_FILE} ...")
final_test_df.to_csv(TEST_OUTPUT_FILE, index=False)

print("✅ 預處理完成！乾淨的 CSV 檔案已生成 (已移除極端值)。")


