import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import ast
import pandas as pd
import numpy as np

# 读取数据集
dataset_name = 'koenshen/EVADE-Bench'
image_test_dataset = datasets.load_dataset(dataset_name, split="image_test")

# 存储统计结果的字典
stats_by_content_type = defaultdict(Counter)

# 遍历数据集的每一行
for row in image_test_dataset:
    content_type = row['content_type']
    single_risk_options_str = row['single_risk_options']

    # 将字符串转换为列表
    try:
        # 尝试使用ast.literal_eval安全地解析字符串
        single_risk_options = ast.literal_eval(single_risk_options_str)

        # 统计每个选项
        if isinstance(single_risk_options, list):
            for option in single_risk_options:
                stats_by_content_type[content_type][option] += 1
    except:
        print(f"Warning: Could not parse single_risk_options: {single_risk_options_str}")
        continue

# 打印统计结果
print("=" * 80)
print("统计结果:")
print("=" * 80)
for content_type, counter in stats_by_content_type.items():
    print(f"\nContent Type: {content_type}")
    # 按字母顺序排序输出
    for option in sorted(counter.keys()):
        print(f"  {option}: {counter[option]}")

# 可视化
# 方案1: 为每个content_type创建单独的子图
content_types = sorted(list(stats_by_content_type.keys()))  # 也对content_type排序
n_content_types = len(content_types)

# 计算子图布局
n_cols = min(3, n_content_types)
n_rows = (n_content_types + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
if n_content_types == 1:
    axes = [axes]
else:
    axes = axes.flatten() if n_content_types > 1 else [axes]

# 为每个content_type绘制柱状图
for idx, content_type in enumerate(content_types):
    ax = axes[idx]
    counter = stats_by_content_type[content_type]

    # 按字母顺序排序选项
    options = sorted(list(counter.keys()))
    counts = [counter[option] for option in options]

    # 绘制柱状图
    bars = ax.bar(range(len(options)), counts, color=plt.cm.Set3(idx % 12))
    ax.set_xlabel('Risk Options', fontsize=10)
    ax.set_ylabel('Sample Count', fontsize=10)
    ax.set_title(f'Content Type: {content_type}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(options)))
    ax.set_xticklabels(options, rotation=45, ha='right', fontsize=8)

    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)

    ax.grid(axis='y', alpha=0.3)

# 隐藏多余的子图
for idx in range(n_content_types, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('single_risk_options_by_content_type.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存为: single_risk_options_by_content_type.png")
plt.show()

# 方案2: 创建一个汇总的热力图
print("\n正在生成热力图...")

# 准备数据用于热力图
all_options = set()
for counter in stats_by_content_type.values():
    all_options.update(counter.keys())

# 按字母顺序排序所有选项
all_options = sorted(list(all_options))
content_types_sorted = sorted(list(stats_by_content_type.keys()))

# 创建矩阵
matrix = []
for content_type in content_types_sorted:
    row = [stats_by_content_type[content_type][option] for option in all_options]
    matrix.append(row)

# 绘制热力图
plt.figure(figsize=(max(12, len(all_options) * 0.8), max(8, len(content_types_sorted) * 0.6)))
sns.heatmap(matrix,
            xticklabels=all_options,
            yticklabels=content_types_sorted,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            cbar_kws={'label': 'Sample Count'})

plt.xlabel('Risk Options', fontsize=12)
plt.ylabel('Content Type', fontsize=12)
plt.title('Distribution of Single Risk Options by Content Type', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('single_risk_options_heatmap.png', dpi=300, bbox_inches='tight')
print(f"热力图已保存为: single_risk_options_heatmap.png")
plt.show()

# 方案3: 创建DataFrame并保存为CSV
print("\n正在保存统计数据到CSV...")
df_data = []
for content_type in sorted(stats_by_content_type.keys()):
    counter = stats_by_content_type[content_type]
    for option in sorted(counter.keys()):
        df_data.append({
            'content_type': content_type,
            'risk_option': option,
            'count': counter[option]
        })

df = pd.DataFrame(df_data)
df.to_csv('single_risk_options_statistics.csv', index=False)
print(f"统计数据已保存为: single_risk_options_statistics.csv")

# 显示汇总信息
print("\n" + "=" * 80)
print("汇总信息:")
print("=" * 80)
print(f"总的 Content Types 数量: {len(stats_by_content_type)}")
print(f"总的 Risk Options 数量: {len(all_options)}")
print(f"所有 Risk Options (按字母顺序): {', '.join(all_options)}")
print(f"总样本数 (计入所有选项): {sum(sum(counter.values()) for counter in stats_by_content_type.values())}")
