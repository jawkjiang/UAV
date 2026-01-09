"""
查看所有takeover攻击的完整参数
"""

import pandas as pd

for attack_type in ['takeover_step', 'takeover_ramp']:
    print("="*80)
    print(f"{attack_type}")
    print("="*80)
    
    df = pd.read_csv(f'../step3_multiModel/output/{attack_type}/cnn/test_attack_info.csv')
    attacked = df[df['attacked'] == True]
    
    if len(attacked) > 0:
        for idx, row in attacked.iterrows():
            print(f"\nFlight {row['flight']}:")
            print(f"  {row['attack_params']}")
    print()
