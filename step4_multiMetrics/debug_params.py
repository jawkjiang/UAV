"""
详细检查参数解析
"""

import pandas as pd
import ast
import re

# 检查drift_ramp
print("="*80)
print("检查 drift_ramp 参数解析")
print("="*80)

df = pd.read_csv('../step3_multiModel/output/drift_ramp/cnn/test_attack_info.csv')
attacked = df[df['attacked'] == True].iloc[0]

params_str = str(attacked['attack_params'])
print(f"\n原始字符串:\n{params_str}\n")

# 清理并解析
params_str_cleaned = re.sub(r'np\.(float64|int64|str_)\((.*?)\)', r'\2', params_str)
print(f"清理后:\n{params_str_cleaned}\n")

try:
    params = ast.literal_eval(params_str_cleaned)
    print("解析成功!")
    print(f"  M = {params.get('M', 'NOT FOUND')}")
    print(f"  T_drift = {params.get('T_drift', 'NOT FOUND')}")
    print(f"  profile = {params.get('profile', 'NOT FOUND')}")
except Exception as e:
    print(f"解析失败: {e}")

# 检查takeover_step
print("\n" + "="*80)
print("检查 takeover_step 参数解析")
print("="*80)

df2 = pd.read_csv('../step3_multiModel/output/takeover_step/cnn/test_attack_info.csv')
attacked2 = df2[df2['attacked'] == True].iloc[0]

params_str2 = str(attacked2['attack_params'])
print(f"\n原始字符串:\n{params_str2}\n")

params_str_cleaned2 = re.sub(r'np\.(float64|int64|str_)\((.*?)\)', r'\2', params_str2)
print(f"清理后:\n{params_str_cleaned2}\n")

try:
    params2 = ast.literal_eval(params_str_cleaned2)
    print("解析成功!")
    print(f"  M = {params2.get('M', 'NOT FOUND')}")
    print(f"  offset_profile = {params2.get('offset_profile', 'NOT FOUND')}")
    print(f"  T_takeover = {params2.get('T_takeover', 'NOT FOUND')}")
    print(f"  duration = {params2.get('duration', 'NOT FOUND')}")
except Exception as e:
    print(f"解析失败: {e}")
