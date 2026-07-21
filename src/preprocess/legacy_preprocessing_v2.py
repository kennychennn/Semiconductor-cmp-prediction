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
# 排除 Label 檔案，避免誤讀
test_files = [f for f in test_files if os.path.basename(f) != os.path.basename(LABEL_PATH1)]

test_df_list = []
for file in test_files:
    df_temp = pd.read_csv(file)
    test_df_list.append(df_temp)
test_big_df = pd.concat(test_df_list, ignore_index=True)
print("測試集資料合併完成。")
if len(test_files) == 0:
    print(f"⚠️ 警告: 在路徑 '{VAL_PATH}' 下找不到任何 CSV 檔案 (已排除標籤檔)。")
    print("請確認路徑是否正確。程式將使用空的 DataFrame 繼續執行，以避免崩潰。")
    # 建立一個空的 DataFrame，欄位與訓練集相同，以防止後續程式碼報錯
    test_big_df = pd.DataFrame(columns=big_df.columns)
else:
    test_df_list = []
    for file in test_files:
        df_temp = pd.read_csv(file)
        test_df_list.append(df_temp)
    test_big_df = pd.concat(test_df_list, ignore_index=True)
    print("測試集資料合併完成。")

# ==========================================
# 3. 繪製所有資料的物理量 vs TIMESTAMP
# ==========================================
if not big_df.empty and 'TIMESTAMP' in big_df.columns:
    print(f"\n正在針對所有資料點進行作圖 (Scatter Plot)...")
    
    # 使用所有資料
    plot_data = big_df.copy()
    
    # 確保 TIMESTAMP 是數值型態並且排序
    plot_data['TIMESTAMP'] = pd.to_numeric(plot_data['TIMESTAMP'], errors='coerce')
    plot_data = plot_data.sort_values(by='TIMESTAMP')
    
    # 找出需要作圖的特徵欄位 (排除 ID, TIMESTAMP 等非機台感測量測數值)
    exclude_cols = ['WAFER_ID', 'TIMESTAMP', 'CHAMBER', 'STAGE', 'DRESSING_WATER_STATUS']
    all_plot_cols = [col for col in plot_data.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(plot_data[col])]
    
    # 根據特徵名稱進行分組
    airbag_cols = [col for col in all_plot_cols if 'AIR_BAG' in col]
    slurry_cols = [col for col in all_plot_cols if 'SLURRY' in col]
    usage_cols = [col for col in all_plot_cols if 'USAGE' in col]
    rotation_cols = [col for col in all_plot_cols if 'ROTATION' in col]
    
    categorized_cols = set(airbag_cols + slurry_cols + usage_cols + rotation_cols)
    other_cols = [col for col in all_plot_cols if col not in categorized_cols]
    
    feature_groups = {
        "Air Bag Variables": airbag_cols,
        "Slurry Variables": slurry_cols,
        "Usage Variables": usage_cols,
        "Rotation Variables": rotation_cols,
        "Other Variables": other_cols
    }
    
    for group_name, cols in feature_groups.items():
        if not cols:
            continue
            
        if group_name == "Air Bag Variables":
            n_cols = 2
        elif group_name == "Slurry Variables":
            n_cols = 1
        else:
            n_cols = 3
        n_rows = max(1, int(np.ceil(len(cols) / n_cols)))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = np.atleast_1d(axes).flatten()  # 確保攤平後可迭代
        
        for i, col in enumerate(cols):
            # 使用散點圖 (Scatter Plot) 來呈現，避免在不同晶圓之間畫線
            axes[i].plot(plot_data['TIMESTAMP'], plot_data[col], marker='.', linestyle='none', markersize=1, color='teal', alpha=0.3)
            axes[i].set_title(f"{col}", fontsize=10, fontweight='bold')
            axes[i].set_xlabel("TIMESTAMP")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, linestyle='--', alpha=0.6)
            
        # 隱藏多餘的空白子圖
        for j in range(len(cols), len(axes)):
            fig.delaxes(axes[j])
            
        plt.suptitle(f"{group_name} vs TIMESTAMP", fontsize=14, fontweight='bold')
        # 調整版面配置，在頂部 (top=0.95) 留出空間給大標題 (suptitle)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=1.5)
        plt.show()
else:
    print("資料集為空或找不到 TIMESTAMP 欄位，無法作圖。")

# ==========================================
# 4. 針對前三個晶圓，繪製「轉速、壓力、研磨液」綜合除錯圖
# ==========================================
if not big_df.empty and 'WAFER_ID' in big_df.columns and 'TIMESTAMP' in big_df.columns:
    # 找出所有不重複的 WAFER_ID 並排序
    unique_wafer_ids = sorted(big_df['WAFER_ID'].unique())
    
    if len(unique_wafer_ids) > 0:
        # 選取前三個晶圓的 ID 以免圖表太多
        target_wafers = unique_wafer_ids[:3]
        print(f"\n正在針對前 {len(target_wafers)} 個晶圓繪製「轉速、壓力、研磨液」綜合時間序列圖...")
        
        for wafer_id in target_wafers:
            # 篩選出特定晶圓的資料
            wafer_df = big_df[big_df['WAFER_ID'] == wafer_id].copy()
            
            # 確保 TIMESTAMP 是數值型態並排序，以便正確繪製序列圖
            wafer_df['TIMESTAMP'] = pd.to_numeric(wafer_df['TIMESTAMP'], errors='coerce')
            wafer_df = wafer_df.sort_values(by='TIMESTAMP')

            # --- 定義研磨狀態 (Polishing Phase) ---
            if 'PRESSURIZED_CHAMBER_PRESSURE' in wafer_df.columns:
                pressure_active = wafer_df['PRESSURIZED_CHAMBER_PRESSURE'] > 0
            else:
                pressure_active = pd.Series(False, index=wafer_df.index)

            # 晶圓或平台任一在轉動
            wafer_is_rotating = wafer_df['WAFER_ROTATION'] > 0.1 if 'WAFER_ROTATION' in wafer_df.columns else False
            stage_is_rotating = wafer_df['STAGE_ROTATION'] > 0.1 if 'STAGE_ROTATION' in wafer_df.columns else False
            
            is_polishing = pressure_active & (wafer_is_rotating | stage_is_rotating)
            
            # 定義旋乾 (Spin-Dry) 階段：晶圓和平台「同時」高速旋轉，但「沒有」壓力
            is_spin_dry = (wafer_is_rotating & stage_is_rotating) & ~pressure_active

            # 建立上下排列的 4 個子圖 (共享 X 軸)
            fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            
            # Plot 1: 轉速 (Motion)
            if 'HEAD_ROTATION' in wafer_df.columns:
                axes[0].plot(wafer_df['TIMESTAMP'], wafer_df['HEAD_ROTATION'], label='HEAD_ROTATION', color='purple', alpha=0.7)
            if 'WAFER_ROTATION' in wafer_df.columns:
                axes[0].plot(wafer_df['TIMESTAMP'], wafer_df['WAFER_ROTATION'], label='WAFER_ROTATION', color='blue', alpha=0.7)
            if 'STAGE_ROTATION' in wafer_df.columns:
                axes[0].plot(wafer_df['TIMESTAMP'], wafer_df['STAGE_ROTATION'], label='STAGE_ROTATION', color='orange', alpha=0.7)
            axes[0].set_ylabel("Rotation Speed")
            axes[0].set_title(f"Wafer ID: {wafer_id} - Motion, Force, and Chemistry")
            axes[0].legend(loc='upper right')
            
            # Plot 2: 壓力 (Force)
            if 'PRESSURIZED_CHAMBER_PRESSURE' in wafer_df.columns:
                axes[1].plot(wafer_df['TIMESTAMP'], wafer_df['PRESSURIZED_CHAMBER_PRESSURE'], label='CHAMBER_PRESSURE', color='red')
            if 'MAIN_OUTER_AIR_BAG_PRESSURE' in wafer_df.columns:
                axes[1].plot(wafer_df['TIMESTAMP'], wafer_df['MAIN_OUTER_AIR_BAG_PRESSURE'], label='AIR_BAG_PRESSURE', color='green', linestyle='--')
            axes[1].set_ylabel("Pressure")
            axes[1].legend(loc='upper right')
            
            # Plot 3: 研磨液 (Chemistry)
            slurry_cols = [c for c in wafer_df.columns if 'SLURRY' in c]
            for c in slurry_cols:
                axes[2].plot(wafer_df['TIMESTAMP'], wafer_df[c], label=c)
            axes[2].set_ylabel("Slurry Flow")
            axes[2].legend(loc='upper right')
            
            # Plot 4: 製程階段 (Chamber)
            if 'CHAMBER' in wafer_df.columns:
                axes[3].plot(wafer_df['TIMESTAMP'], wafer_df['CHAMBER'], label='CHAMBER', color='black', drawstyle='steps-post', linewidth=2)
            axes[3].set_ylabel("Chamber")
            axes[3].set_xlabel("TIMESTAMP")
            axes[3].legend(loc='upper right')

            # 為所有子圖畫上背景色
            for ax in axes:
                ymin, ymax = ax.get_ylim()
                ax.fill_between(wafer_df['TIMESTAMP'], ymin, ymax, where=is_polishing, color='gold', alpha=0.3, label='Polishing' if ax == axes[0] else "")
                ax.fill_between(wafer_df['TIMESTAMP'], ymin, ymax, where=is_spin_dry, color='cyan', alpha=0.4, label='Spin-Dry' if ax == axes[0] else "")
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_ylim(ymin, ymax)
                
            # 避免標籤重複
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[0].legend(by_label.values(), by_label.keys(), loc='upper left')
            
            plt.tight_layout(pad=1.5)
            plt.show()
                
    else:
        print("警告: 資料集中找不到任何晶圓的資料。")
else:
    print("資料集為空或找不到 WAFER_ID/TIMESTAMP 欄位，無法繪製特定晶圓的圖表。")

# ==========================================
# 新增區塊：檢查氣囊壓力變數之間的共線性
# ==========================================
print("\n正在檢查氣囊壓力變數之間的相關性 (共線性檢查)...")
pressure_cols_to_check = [
    'MAIN_OUTER_AIR_BAG_PRESSURE',
    'CENTER_AIR_BAG_PRESSURE',
    'RIPPLE_AIR_BAG_PRESSURE',
    'EDGE_AIR_BAG_PRESSURE'
]

# 確保這些欄位都存在於 DataFrame 中
existing_pressure_cols = [col for col in pressure_cols_to_check if col in big_df.columns]

if len(existing_pressure_cols) > 1:
    # 計算相關係數矩陣
    pressure_corr_matrix = big_df[existing_pressure_cols].corr()
    
    print("氣囊壓力相關係數矩陣:")
    print(pressure_corr_matrix)
    
    # 繪製熱力圖
    plt.figure(figsize=(8, 6))
    sns.heatmap(pressure_corr_matrix, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5)
    plt.title('Correlation Matrix of Air Bag Pressures', fontsize=14)
    plt.show()
