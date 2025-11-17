import json


def extract_mark_result(json_str):
    """从标注结果字符串中提取markResult"""
    try:
        # 解析JSON字符串
        mark_data = json.loads(json_str)[0]
        # 获取markResult并解析
        mark_result = json.loads(mark_data['markResult'])
        return mark_result
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def process_jsonl_file(file_path):
    # 初始化三个列表来存储每个标注人的结果
    annotator1_results = []
    annotator2_results = []
    annotator3_results = []

    # 读取JSONL文件
    with open(file_path, 'r', encoding='gbk') as file:
        for line in file:
            try:
                # 解析每一行的JSON数据
                data = json.loads(line.strip())

                # 获取ID作为标识
                sample_id = data['id']

                # 提取三个标注人的结果
                result1 = extract_mark_result(data['标注环节结果1'])
                result2 = extract_mark_result(data['标注环节结果2'])
                result3 = extract_mark_result(data['标注环节结果3'])

                # 将结果添加到对应的列表中
                if result1:
                    annotator1_results.append({
                        'id': sample_id,
                        'annotator': data['标注环节人员1'],
                        'result': result1
                    })
                if result2:
                    annotator2_results.append({
                        'id': sample_id,
                        'annotator': data['标注环节人员2'],
                        'result': result2
                    })
                if result3:
                    annotator3_results.append({
                        'id': sample_id,
                        'annotator': data['标注环节人员3'],
                        'result': result3
                    })

            except json.JSONDecodeError:
                print(f"Error parsing line: {line[:100]}...")
                continue
            except KeyError as e:
                print(f"Missing key in data: {e}")
                continue

    return annotator1_results, annotator2_results, annotator3_results


# 使用示例
file_path = '../data/Evade-疾病标注-文本_GBK__20251117100131.jsonl'
results1, results2, results3 = process_jsonl_file(file_path)

# 打印结果示例
print(f"标注人员1 ({results1[0]['annotator']}) 的标注数量: {len(results1)}")
print(f"标注人员2 ({results2[0]['annotator']}) 的标注数量: {len(results2)}")
print(f"标注人员3 ({results3[0]['annotator']}) 的标注数量: {len(results3)}")

# 打印第一个样本的标注结果示例
print("\n第一个样本的标注结果:")
print(f"标注人员1: {results1[0]['result']}")
print(f"标注人员2: {results2[0]['result']}")
print(f"标注人员3: {results3[0]['result']}")
