import json
import numpy as np
import re


# ------------------------------------------------------------
#                1. 正确提取标签 A~Z
# ------------------------------------------------------------
def extract_labels(json_str):
    """从 markResult 文本列表中提取 A~Z 标签"""
    try:
        arr = json.loads(json_str)  # 外层 list
        mark_result = json.loads(arr[0]["markResult"])  # 内层 list of text

        labels = []
        for item in mark_result:
            # m = re.match(r'^\s*-\s*([A-Z])\.', item.strip()) # 只能计算字母前有-的
            m = re.match(r'^\s*-?\s*([A-Z])\.', item.strip()) # 更通用

            if m:
                labels.append(m.group(1))

        return sorted(labels)  # 返回排序后的标签，便于比较

    except Exception as e:
        return None


# ------------------------------------------------------------
#                2. 读取 JSONL，构建 3 位标注者 dict
# 同时读取 ground_truth
# ------------------------------------------------------------
def process_jsonl_files_to_dicts(file_paths):
    """
    处理一个或多个jsonl文件，汇总标注结果和ground truth

    Args:
        file_paths: 单个文件路径(str)或多个文件路径列表(list)

    Returns:
        ((dict1, name1), (dict2, name2), (dict3, name3), ground_truth_dict)
    """
    # 统一处理输入：如果是字符串则转为列表
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    elif not isinstance(file_paths, list):
        raise TypeError("file_paths 必须是字符串或列表")

    dict1, dict2, dict3 = {}, {}, {}
    ground_truth_dict = {}
    name1 = name2 = name3 = None

    # 遍历所有文件
    for file_path in file_paths:
        print(f"正在处理文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        sid = data['id']

                        # 提取 ground truth
                        gt_str = data.get('single_risk_options', '[]')
                        gt_list = eval(gt_str)  # 将字符串转换为列表
                        ground_truth_dict[sid] = sorted(gt_list)

                        r1 = extract_labels(data.get('标注环节结果1'))
                        r2 = extract_labels(data.get('标注环节结果2'))
                        r3 = extract_labels(data.get('标注环节结果3'))

                        if r1 is not None:
                            dict1[sid] = r1
                            if name1 is None:
                                name1 = data.get("标注环节人员1")

                        if r2 is not None:
                            dict2[sid] = r2
                            if name2 is None:
                                name2 = data.get("标注环节人员2")

                        if r3 is not None:
                            dict3[sid] = r3
                            if name3 is None:
                                name3 = data.get("标注环节人员3")

                    except Exception as e:
                        print(f"处理行时出错: {e}")
                        continue
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
            continue

    print(f"文件处理完成！共处理 {len(file_paths)} 个文件")
    return (dict1, name1), (dict2, name2), (dict3, name3), ground_truth_dict


# ------------------------------------------------------------
#                3. 计算完全一致率和部分一致率
# ------------------------------------------------------------
def calculate_accuracy(annotator_dict, annotator_name, ground_truth_dict):
    """
    计算一个标注者的完全一致率和部分一致率

    完全一致率：标注结果集合 == ground_truth 集合
    部分一致率：标注结果集合 与 ground_truth 集合有交集
    """
    full_match_count = 0
    partial_match_count = 0
    total_count = 0

    for sid, annotated_labels in annotator_dict.items():
        if sid not in ground_truth_dict:
            continue

        gt_labels = ground_truth_dict[sid]
        annotated_set = set(annotated_labels)
        gt_set = set(gt_labels)

        # 完全一致：两个集合相等
        if annotated_set == gt_set:
            full_match_count += 1

        # 部分一致：两个集合有交集
        if len(annotated_set.intersection(gt_set)) > 0:
            partial_match_count += 1

        total_count += 1

    full_accuracy = full_match_count / total_count if total_count > 0 else 0
    partial_accuracy = partial_match_count / total_count if total_count > 0 else 0

    return {
        'name': annotator_name,
        'full_accuracy': full_accuracy,
        'partial_accuracy': partial_accuracy,
        'total_count': total_count,
        'full_match_count': full_match_count,
        'partial_match_count': partial_match_count
    }


# ------------------------------------------------------------
#                4. 主函数
# ------------------------------------------------------------
if __name__ == "__main__":
    file_paths = [
        # "../data/Evade-增高标注-文本_GBK__20251117100122.jsonl",
        # "../data/Evade-疾病标注-文本_GBK__20251117100131.jsonl",
        "../data/Evade-增高标注-图像_GBK__20251117100128.jsonl",
        "../data/Evade-疾病标注-图像_GBK__20251117100345.jsonl",
    ]

    (dict1, name1), (dict2, name2), (dict3, name3), ground_truth_dict = process_jsonl_files_to_dicts(file_paths)

    print("=" * 80)
    print("数据加载完成")
    print("=" * 80)
    print(f"{name1}: {len(dict1)} annotations")
    print(f"{name2}: {len(dict2)} annotations")
    print(f"{name3}: {len(dict3)} annotations")
    print(f"Ground Truth: {len(ground_truth_dict)} samples")
    print()

    # 计算每个标注者的准确率
    result1 = calculate_accuracy(dict1, name1, ground_truth_dict)
    result2 = calculate_accuracy(dict2, name2, ground_truth_dict)
    result3 = calculate_accuracy(dict3, name3, ground_truth_dict)

    print("=" * 80)
    print("各标注人员的准确率统计")
    print("=" * 80)

    # 打印详细结果
    for result in [result1, result2, result3]:
        print(f"\n标注人员: {result['name']}")
        print(f"  总样本数: {result['total_count']}")
        print(f"  完全一致数: {result['full_match_count']}")
        print(f"  部分一致数: {result['partial_match_count']}")
        print(f"  完全一致率: {result['full_accuracy']:.4f} ({result['full_accuracy'] * 100:.2f}%)")
        print(f"  部分一致率: {result['partial_accuracy']:.4f} ({result['partial_accuracy'] * 100:.2f}%)")

    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)

    # 计算平均值
    avg_full_accuracy = (result1['full_accuracy'] + result2['full_accuracy'] + result3['full_accuracy']) / 3
    avg_partial_accuracy = (result1['partial_accuracy'] + result2['partial_accuracy'] + result3['partial_accuracy']) / 3

    print(f"\n3人完全一致率平均值: {avg_full_accuracy:.4f} ({avg_full_accuracy * 100:.2f}%)")
    print(f"3人部分一致率平均值: {avg_partial_accuracy:.4f} ({avg_partial_accuracy * 100:.2f}%)")

    print("\n" + "=" * 80)
    print("8个关键指标汇总")
    print("=" * 80)
    print(f"1. {name1} 完全一致率: {result1['full_accuracy']:.4f}")
    print(f"2. {name1} 部分一致率: {result1['partial_accuracy']:.4f}")
    print(f"3. {name2} 完全一致率: {result2['full_accuracy']:.4f}")
    print(f"4. {name2} 部分一致率: {result2['partial_accuracy']:.4f}")
    print(f"5. {name3} 完全一致率: {result3['full_accuracy']:.4f}")
    print(f"6. {name3} 部分一致率: {result3['partial_accuracy']:.4f}")
    print(f"7. 3人完全一致率平均值: {avg_full_accuracy:.4f}")
    print(f"8. 3人部分一致率平均值: {avg_partial_accuracy:.4f}")
