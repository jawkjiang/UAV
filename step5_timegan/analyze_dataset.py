"""
对比分析TimeGAN扩充前后的数据集质量
检查可能导致梯度爆炸的数据问题
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('../step3b_multiModelGeneral')

from feature_engineering import compute_all_features, get_feature_columns


def analyze_dataframe(df, name="Dataset"):
    """分析数据集的统计特性"""
    print(f"\n{'='*80}")
    print(f"{name} 数据质量分析")
    print(f"{'='*80}")
    
    print(f"\n基本信息:")
    print(f"  总记录数: {len(df):,}")
    print(f"  飞行数: {df['flight'].nunique()}")
    print(f"  列数: {len(df.columns)}")
    
    # 检查NaN/Inf
    print(f"\n缺失值/异常值检查:")
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"  ⚠️  包含NaN的列: {nan_cols}")
        for col in nan_cols:
            nan_count = df[col].isna().sum()
            nan_pct = nan_count / len(df) * 100
            print(f"     - {col}: {nan_count:,} ({nan_pct:.2f}%)")
    else:
        print(f"  ✓ 无NaN值")
    
    # 检查无穷值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_found = False
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            if not inf_found:
                print(f"  ⚠️  包含Inf的列:")
                inf_found = True
            print(f"     - {col}: {inf_count:,}")
    if not inf_found:
        print(f"  ✓ 无Inf值")
    
    # 数值范围
    print(f"\n数值特征统计 (min, max, mean, std):")
    for col in numeric_cols:
        if col not in ['flight', 'label', 'is_attacked']:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:20s}: [{vals.min():10.4f}, {vals.max():10.4f}], "
                      f"mean={vals.mean():10.4f}, std={vals.std():10.4f}")
                
                # 检查极端值
                if abs(vals.min()) > 1e6 or abs(vals.max()) > 1e6:
                    print(f"    ⚠️  极端值范围!")
                if vals.std() > 1e6:
                    print(f"    ⚠️  标准差过大!")
    
    # 标签分布
    if 'label' in df.columns:
        print(f"\n标签分布:")
        label_dist = df['label'].value_counts()
        print(f"  正样本: {label_dist.get(1, 0):,} ({df['label'].mean()*100:.2f}%)")
        print(f"  负样本: {label_dist.get(0, 0):,} ({(1-df['label'].mean())*100:.2f}%)")
    
    return df


def compare_datasets(original_path, timegan_path):
    """对比原始数据和TimeGAN增强数据"""
    print("\n" + "="*80)
    print("数据集对比分析：原始 vs TimeGAN增强")
    print("="*80)
    
    # 读取数据
    print("\n加载数据集...")
    orig_df = pd.read_csv(original_path, low_memory=False)
    tgan_df = pd.read_csv(timegan_path, low_memory=False)
    
    print(f"  原始数据: {original_path}")
    print(f"  TimeGAN数据: {timegan_path}")
    
    # 分析原始数据
    orig_df = analyze_dataframe(orig_df, "原始数据集")
    
    # 分析TimeGAN数据
    tgan_df = analyze_dataframe(tgan_df, "TimeGAN增强数据集")
    
    # 对比特征分布
    print(f"\n{'='*80}")
    print("特征分布对比 (原始 vs TimeGAN)")
    print(f"{'='*80}")
    
    numeric_cols = orig_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols 
                    if col not in ['flight', 'label', 'is_attacked']]
    
    print(f"\n{'特征名':<20} | {'原始均值':<12} | {'TG均值':<12} | {'原始std':<12} | {'TG std':<12} | 差异")
    print("-" * 95)
    
    for col in feature_cols:
        if col in orig_df.columns and col in tgan_df.columns:
            orig_mean = orig_df[col].mean()
            tgan_mean = tgan_df[col].mean()
            orig_std = orig_df[col].std()
            tgan_std = tgan_df[col].std()
            
            mean_diff = abs(tgan_mean - orig_mean) / (abs(orig_mean) + 1e-7)
            std_diff = abs(tgan_std - orig_std) / (abs(orig_std) + 1e-7)
            
            flag = ""
            if mean_diff > 1.0 or std_diff > 1.0:
                flag = "⚠️ 分布差异大"
            elif np.isnan(tgan_mean) or np.isnan(tgan_std):
                flag = "❌ NaN出现"
            elif np.isinf(tgan_mean) or np.isinf(tgan_std):
                flag = "❌ Inf出现"
            
            print(f"{col:<20} | {orig_mean:>11.4f} | {tgan_mean:>11.4f} | "
                  f"{orig_std:>11.4f} | {tgan_std:>11.4f} | {flag}")
    
    # 检查是否有新增的异常值
    print(f"\n{'='*80}")
    print("异常值统计对比")
    print(f"{'='*80}")
    
    for col in feature_cols:
        if col in orig_df.columns and col in tgan_df.columns:
            orig_outliers = ((orig_df[col] > orig_df[col].quantile(0.99)) | 
                           (orig_df[col] < orig_df[col].quantile(0.01))).sum()
            tgan_outliers = ((tgan_df[col] > tgan_df[col].quantile(0.99)) | 
                           (tgan_df[col] < tgan_df[col].quantile(0.01))).sum()
            
            if tgan_outliers > orig_outliers * 1.5:
                print(f"  ⚠️  {col}: 异常值增加 {orig_outliers} -> {tgan_outliers}")


if __name__ == "__main__":
    # 对比训练集
    print("\n" + "="*80)
    print("对比训练集")
    print("="*80)
    
    # 需要你提供原始数据路径 (step3b的训练集)
    original_train = Path("../step3b_multiModelGeneral/output/train_with_attacks.csv")
    timegan_train = Path("output/train_with_attacks.csv")
    
    if original_train.exists() and timegan_train.exists():
        compare_datasets(original_train, timegan_train)
    else:
        if not original_train.exists():
            print(f"⚠️  原始训练集不存在: {original_train}")
        if not timegan_train.exists():
            print(f"⚠️  TimeGAN训练集不存在: {timegan_train}")
        
        # 至少分析TimeGAN数据集
        if timegan_train.exists():
            print("\n仅分析TimeGAN数据集:")
            df = pd.read_csv(timegan_train, low_memory=False)
            analyze_dataframe(df, "TimeGAN训练集")
