"""
详细解释：FP窗口数 vs 误报次数的关系
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

print("=" * 100)
print("FP窗口数 vs 误报次数的关系详解".center(100))
print("=" * 100)

print("""
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 核心概念：窗口聚合机制 (Window Aggregation)                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  【窗口级别】 vs 【事件级别】                                                                 │
│                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐                       │
│  │ 时间线：每个方块代表一个滑动窗口 (0.05秒间隔)                     │                       │
│  ├─────────────────────────────────────────────────────────────────┤                       │
│  │                                                                 │                       │
│  │  [N][N][N][F][F][F][F][N][N][F][N][N][N][F][F][N][N]           │                       │
│  │           └───┬───┘        │              └─┬─┘                 │                       │
│  │         误报事件1       误报事件2        误报事件3                │                       │
│  │                                                                 │                       │
│  │  说明：                                                          │                       │
│  │  - N = 正常窗口被正确预测 (TN)                                   │                       │
│  │  - F = 正常窗口被误报为攻击 (FP)                                 │                       │
│  │                                                                 │                       │
│  │  统计：                                                          │                       │
│  │  - FP窗口总数: 9个                                               │                       │
│  │  - 误报事件数: 3次（连续的FP窗口聚合为1次事件）                   │                       │
│  └─────────────────────────────────────────────────────────────────┘                       │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
""")

print("\n【聚合规则：从窗口到事件】")
print("-" * 100)

print("""
代码逻辑（来自 time_aware_metrics.py 的 calculate_mtbfa 函数）：

```python
# 统计此飞行的误报事件（连续误报算作一次）
in_false_alarm = False
for i in range(len(flight_y_true)):
    if flight_y_true[i] == 0 and flight_y_pred[i] == 1:  # False Positive窗口
        if not in_false_alarm:
            total_false_alarms += 1  # 开始一次新的误报事件
            in_false_alarm = True
    elif flight_y_true[i] == 0:  # 正常窗口被正确预测
        in_false_alarm = False  # 结束当前误报事件
    else:  # 攻击窗口
        in_false_alarm = False
```

聚合规则：
  1. 遇到第一个FP窗口 → 误报事件计数+1，标记 in_false_alarm=True
  2. 后续连续的FP窗口 → 不增加计数（仍然是同一次误报事件）
  3. 遇到TN窗口或攻击窗口 → 结束当前误报事件，重置 in_false_alarm=False
  4. 再次遇到FP窗口 → 开始新的误报事件，计数+1
""")

print("\n\n【实际案例：Transformer的107个FP窗口 → 18次误报】")
print("-" * 100)

print("""
假设场景（基于实际数据推断）：

测试集情况：
  - 总窗口数: 20,302个
  - 正常窗口: 18,187个 (TN=18,080, FP=107)
  - 攻击窗口: 2,115个 (TP=2,024, FN=91)
  - 正常飞行数: 约157个（209总飞行 - 52攻击飞行）

FP分布模式（示例）：

飞行 #1 (正常):  [TN][TN][TN][FP][FP][TN][TN]... → 1次误报，包含2个FP窗口
飞行 #2 (正常):  [TN][TN][TN][TN][TN][TN][TN]... → 0次误报
飞行 #3 (正常):  [TN][FP][TN][TN][FP][FP][FP][TN] → 2次误报，包含4个FP窗口
飞行 #4 (正常):  [TN][TN][TN][FP][TN][TN][TN]... → 1次误报，包含1个FP窗口
...
总计: 18次误报事件，107个FP窗口

平均每次误报事件包含: 107 / 18 ≈ 5.9个窗口
每个窗口时长: 0.05秒
平均每次误报持续时间: 5.9 × 0.05 ≈ 0.3秒
""")

print("\n\n【关键洞察】")
print("=" * 100)

print("""
1. **为什么要聚合？**
   - 滑动窗口有重叠，连续的FP窗口实际上检测的是同一个"误报事件"
   - 对于实际应用，飞行员只会听到一次警报声，不是N次
   - MTBFA关心的是"警报频率"，不是"误判窗口数"

2. **聚合比例因素**
   ┌────────────────────────────────────────────────────────────┐
   │ 影响因素                    对聚合比例的影响                │
   ├────────────────────────────────────────────────────────────┤
   │ 窗口重叠度 ↑               同一事件包含更多窗口 ↑          │
   │ 误报持续时间 ↑             同一事件包含更多窗口 ↑          │
   │ 误报分散程度 ↑             误报事件数增加 ↑                │
   │ 模型稳定性 ↑               误报更集中，事件数 ↓            │
   └────────────────────────────────────────────────────────────┘

3. **Transformer的具体情况**
   - 107个FP窗口 → 18次误报事件
   - 聚合比例: 107/18 ≈ 6:1
   - 说明：Transformer的误报通常持续约0.3秒（6个窗口）
   - 这可能是因为Transformer在某些正常模式下会短暂地误判

4. **对比GRU**
   - 145个FP窗口 → 11次误报事件
   - 聚合比例: 145/11 ≈ 13:1
   - 说明：GRU的误报虽然窗口数更多，但更集中
   - 每次误报持续约0.65秒（13个窗口），但事件数更少
   - 这意味着GRU的误报更"稳定"，一旦误报就持续一段时间
   - 但从用户体验看，误报次数少更重要！

5. **MTBFA的计算**
   
   MTBFA = 正常飞行总时长 / 误报事件数
   
   Transformer:
     正常飞行总时长 ≈ 18 × 0.219小时 ≈ 3.94小时
     误报事件数 = 18次
     MTBFA = 3.94 / 18 = 0.219小时 = 13.1分钟
   
   GRU:
     正常飞行总时长 ≈ 11 × 0.358小时 ≈ 3.94小时（相同测试集）
     误报事件数 = 11次
     MTBFA = 3.94 / 11 = 0.358小时 = 21.5分钟

┌─────────────────────────────────────────────────────────────────────────────┐
│ 总结：FP窗口数 vs 误报次数                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  关系：FP窗口数 ≥ 误报次数（连续FP窗口聚合为1次误报事件）                    │
│                                                                             │
│  ┌────────────┬─────────────┬─────────────┬──────────┬──────────┐         │
│  │   模型     │  FP窗口数   │  误报次数   │ 聚合比例 │  MTBFA   │         │
│  ├────────────┼─────────────┼─────────────┼──────────┼──────────┤         │
│  │ Transformer│    107      │     18      │   6:1    │ 13.1分钟 │         │
│  │    TCN     │     79      │     18      │   4:1    │ 13.1分钟 │         │
│  │    GRU     │    145      │     11      │  13:1    │ 21.5分钟 │ ⭐      │
│  │   LSTM     │    145      │     11      │  13:1    │ 21.5分钟 │         │
│  │    CNN     │     ?       │     15      │   ?      │ 15.8分钟 │         │
│  └────────────┴─────────────┴─────────────┴──────────┴──────────┘         │
│                                                                             │
│  关键发现：                                                                  │
│  • 相同的FP窗口数可能对应不同的误报次数（取决于分布）                        │
│  • 更少的误报次数 = 更好的用户体验                                           │
│  • MTBFA直接衡量误报频率，比Precision更实用！                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("=" * 100)
print("解释完成！".center(100))
print("=" * 100)

# 创建可视化图
print("\n正在生成可视化图...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 子图1: 时间线示例
ax = axes[0]
ax.set_xlim(0, 20)
ax.set_ylim(0, 3)
ax.axis('off')
ax.set_title('FP窗口聚合为误报事件的示例', fontsize=14, fontweight='bold', pad=20)

# 绘制窗口
window_states = ['N','N','N','F','F','F','F','N','N','F','N','N','N','F','F','N','N','N','N','N']
colors = {'N': '#51CF66', 'F': '#FF6B6B'}
labels = {'N': 'TN (正常)', 'F': 'FP (误报)'}

y_pos = 2.0
for i, state in enumerate(window_states):
    rect = Rectangle((i, y_pos), 0.9, 0.5, 
                     facecolor=colors[state], 
                     edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(i + 0.45, y_pos + 0.25, state, 
           ha='center', va='center', fontsize=10, fontweight='bold')

# 标注误报事件
event1_x = [3, 6]
event2_x = [9, 9]
event3_x = [13, 14]

# 事件1
ax.plot([event1_x[0]+0.45, event1_x[1]+0.45], [y_pos-0.3, y_pos-0.3], 
       'r-', linewidth=3, alpha=0.7)
ax.text((event1_x[0]+event1_x[1]+0.9)/2, y_pos-0.5, '误报事件1\n(4个窗口)', 
       ha='center', va='top', fontsize=9, color='red', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

# 事件2
ax.plot([event2_x[0]+0.45, event2_x[1]+0.45], [y_pos-0.3, y_pos-0.3], 
       'r-', linewidth=3, alpha=0.7)
ax.text((event2_x[0]+event2_x[1]+0.9)/2, y_pos-0.5, '误报事件2\n(1个窗口)', 
       ha='center', va='top', fontsize=9, color='red', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

# 事件3
ax.plot([event3_x[0]+0.45, event3_x[1]+0.45], [y_pos-0.3, y_pos-0.3], 
       'r-', linewidth=3, alpha=0.7)
ax.text((event3_x[0]+event3_x[1]+0.9)/2, y_pos-0.5, '误报事件3\n(2个窗口)', 
       ha='center', va='top', fontsize=9, color='red', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

# 统计信息
ax.text(10, 0.5, 
       f'统计：\nFP窗口总数: 7个\n误报事件数: 3次\n聚合比例: 7:3 ≈ 2.3:1', 
       ha='center', va='center', fontsize=11, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))

# 图例
green_patch = mpatches.Patch(color='#51CF66', label='TN: 正常窗口被正确预测')
red_patch = mpatches.Patch(color='#FF6B6B', label='FP: 正常窗口被误报为攻击')
ax.legend(handles=[green_patch, red_patch], loc='upper left', fontsize=10)

# 子图2: 模型对比
ax = axes[1]
models = ['Transformer', 'TCN', 'GRU', 'LSTM']
fp_windows = [107, 79, 145, 145]
false_alarms = [18, 18, 11, 11]
mtbfa_minutes = [13.1, 13.1, 21.5, 21.5]

x = np.arange(len(models))
width = 0.25

# FP窗口数
bars1 = ax.bar(x - width, fp_windows, width, label='FP窗口数', 
              color='#FF6B6B', alpha=0.7, edgecolor='black')

# 误报次数
bars2 = ax.bar(x, false_alarms, width, label='误报事件数', 
              color='#FF8787', alpha=0.7, edgecolor='black')

# MTBFA
bars3 = ax.bar(x + width, mtbfa_minutes, width, label='MTBFA (分钟)', 
              color='#FFA94D', alpha=0.7, edgecolor='black')

# 标注数值
for i, (b1, b2, b3) in enumerate(zip(bars1, bars2, bars3)):
    ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 2,
           f'{fp_windows[i]}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 2,
           f'{false_alarms[i]}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(b3.get_x() + b3.get_width()/2, b3.get_height() + 2,
           f'{mtbfa_minutes[i]:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('数值', fontsize=12, fontweight='bold')
ax.set_xlabel('模型', fontsize=12, fontweight='bold')
ax.set_title('各模型的FP窗口数、误报次数和MTBFA对比', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('paper_figures/fp_windows_vs_false_alarms.png', dpi=300, bbox_inches='tight')
print(f"✓ 可视化图已保存: paper_figures/fp_windows_vs_false_alarms.png")

print("\n" + "=" * 100)
