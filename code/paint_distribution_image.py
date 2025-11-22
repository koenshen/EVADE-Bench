import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import ast
import pandas as pd
import numpy as np

# 读取数据集
dataset_name = 'koenshen/EVADE-Bench'
image_test_dataset = datasets.load_dataset(dataset_name, split="text")

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
    print(f"\n{content_type}")
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

    # ax.grid(axis='y', alpha=0.3)

    # 去除网格线
    ax.grid(False)

    # 只保留左边框和下边框，去除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # 设置y轴从0开始，确保即使所有值都是0也能看到
    ax.set_ylim(bottom=0)

# 隐藏多余的子图
for idx in range(n_content_types, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('single_risk_options_by_content_type.png', dpi=100, bbox_inches='tight')
print(f"\n图表已保存为: single_risk_options_by_content_type.png")
plt.show()

# 显示汇总信息
print("\n" + "=" * 80)
print("汇总信息:")
print("=" * 80)
print(f"总的 Content Types 数量: {len(stats_by_content_type)}")
print(f"总样本数 (计入所有选项): {sum(sum(counter.values()) for counter in stats_by_content_type.values())}")


# 新增函数：生成并打印markdown表格
def print_markdown_table(stats_by_content_type):
    """
    生成并打印markdown格式的统计表格

    参数:
        stats_by_content_type: 按content_type统计的字典
    """
    # 获取所有出现过的risk options（A-Z的字母）
    all_options = set()
    for counter in stats_by_content_type.values():
        all_options.update(counter.keys())

    # 按字母顺序排序
    sorted_options = sorted(list(all_options))

    # 获取所有content_type并排序
    sorted_content_types = sorted(list(stats_by_content_type.keys()))

    # 构建表头
    header = "| Type | " + " | ".join(sorted_options) + " |"
    separator = "|------|" + "|".join(["------" for _ in sorted_options]) + "|"

    # 打印markdown表格
    print("\n" + "=" * 80)
    print("Markdown 统计表格:")
    print("=" * 80)
    print(header)
    print(separator)

    # 为每个content_type打印一行
    for content_type in sorted_content_types:
        counter = stats_by_content_type[content_type]
        row = f"| {content_type} |"

        for option in sorted_options:
            count = counter.get(option, 0)  # 如果该选项不存在，返回0
            row += f" {count} |"

        print(row)

    print("\n")

    # 可选：同时返回pandas DataFrame，方便进一步处理
    data = []
    for content_type in sorted_content_types:
        counter = stats_by_content_type[content_type]
        row_data = {'Type': content_type}
        for option in sorted_options:
            row_data[option] = counter.get(option, 0)
        data.append(row_data)

    df = pd.DataFrame(data)
    return df


# 调用函数生成markdown表格
df_result = print_markdown_table(stats_by_content_type)