import json
import numpy as np
import re
from itertools import combinations


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
            m = re.match(r'^\s*-\s*([A-Z])\.', item.strip())
            if m:
                labels.append(m.group(1))

        return labels  # 返回 ['G','S'] 等

    except Exception as e:
        return None


# ------------------------------------------------------------
#                2. 读取 JSONL，构建 3 位标注者 dict
# ------------------------------------------------------------
def process_jsonl_files_to_dicts_simple(file_paths):
    """
    处理一个或多个jsonl文件，汇总3个标注者的标注结果

    Args:
        file_paths: 单个文件路径(str)或多个文件路径列表(list)

    Returns:
        ((dict1, name1), (dict2, name2), (dict3, name3))
    """
    # 统一处理输入：如果是字符串则转为列表
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    elif not isinstance(file_paths, list):
        raise TypeError("file_paths 必须是字符串或列表")

    dict1, dict2, dict3 = {}, {}, {}
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
                        continue
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
            continue

    print(f"文件处理完成！共处理 {len(file_paths)} 个文件")
    return (dict1, name1), (dict2, name2), (dict3, name3)


# ------------------------------------------------------------
#           3. multi-hot 编码标签 ['G','S'] -> 26维向量
# ------------------------------------------------------------
def encode_labels_multi_hot(label_list, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    vec = np.zeros(len(alphabet), dtype=np.int32)
    for l in label_list:
        if l in alphabet:
            vec[alphabet.index(l)] = 1
    return vec


# ------------------------------------------------------------
#           4. 三人交集样本构建矩阵
# ------------------------------------------------------------
def build_alpha_matrix_intersection(dict1, dict2, dict3):
    common_ids = sorted(set(dict1.keys()) & set(dict2.keys()) & set(dict3.keys()))
    print(f"Three annotators all labeled: {len(common_ids)} samples")

    matrices = []
    for sid in common_ids:
        v1 = encode_labels_multi_hot(dict1[sid])
        v2 = encode_labels_multi_hot(dict2[sid])
        v3 = encode_labels_multi_hot(dict3[sid])
        matrices.append(np.stack([v1, v2, v3], axis=0))

    return np.stack(matrices, axis=0)


# ------------------------------------------------------------
#                5. Krippendorff’s Alpha
# ------------------------------------------------------------
def krippendorffs_alpha(matrix):
    N, k, d = matrix.shape

    # Do
    Do = 0
    count = 0
    for i in range(N):
        for a, b in combinations(range(k), 2):
            Do += np.sum(matrix[i,a] != matrix[i,b])
            count += 1
    Do /= count

    # De
    flat = matrix.reshape(-1, d)
    M = len(flat)
    De = 0
    count = 0
    for a, b in combinations(range(M), 2):
        De += np.sum(flat[a] != flat[b])
        count += 1
    De /= count

    return 1 - Do / De


# ------------------------------------------------------------
#                6. 主程序
# ------------------------------------------------------------
if __name__ == "__main__":
    file_paths = [
        "../data/Evade-增高标注-文本_GBK__20251117100122.jsonl",
        "../data/Evade-疾病标注-文本_GBK__20251117100131.jsonl",
    ]

    (dict1, name1), (dict2, name2), (dict3, name3) = process_jsonl_files_to_dicts_simple(file_paths)

    print(f"{name1}: {len(dict1)} annotations")
    print(f"{name2}: {len(dict2)} annotations")
    print(f"{name3}: {len(dict3)} annotations")

    alpha_input = build_alpha_matrix_intersection(dict1, dict2, dict3)

    alpha = krippendorffs_alpha(alpha_input)
    print("Krippendorff’s Alpha:", alpha)

    # Debug: 打印 Label 分布
    flat = alpha_input.reshape(-1, 26)
    freq = flat.sum(axis=0)
    print("\nLabel frequencies:")
    for i, c in enumerate(freq):
        print(f"{chr(65+i)}: {c}")
