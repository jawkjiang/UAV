import pandas as pd

# 检查合成数据
print("=" * 80)
print("检查合成数据")
print("=" * 80)
df_synth = pd.read_csv('output/synthetic_normal_flights.csv', nrows=1000)
print('列数:', len(df_synth.columns))
print('列名:', list(df_synth.columns))
print('\ndelta_t在列中?', 'delta_t' in df_synth.columns)
if 'delta_t' in df_synth.columns:
    print('delta_t统计:', df_synth['delta_t'].describe())
else:
    print('⚠️ delta_t不存在！')

# 检查训练数据
print("\n" + "=" * 80)
print("检查训练数据")
print("=" * 80)
df = pd.read_csv('output/train_with_attacks.csv', nrows=2000, low_memory=False)
print('总列数:', len(df.columns))
print('\n前20列:', list(df.columns)[:20])

print('\nNaN检查:')
for col in df.columns:
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        print(f'  {col}: {nan_count} ({nan_count/len(df)*100:.1f}%)')

