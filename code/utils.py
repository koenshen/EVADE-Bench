import os
import shutil
import json
import random
import re
from pyexpat.errors import messages

from openai import OpenAI
import datasets
import base64
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict
import time
from datetime import datetime
import difflib
import ast
from typing import ByteString
import math
import copy
from PIL import Image
from copy import deepcopy
import threading
from queue import Queue
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_token_diff(texta, textb):
    """
    计算两个字符串之间的差异，返回 textb 相对于 texta 的不同部分。
    """
    # 使用 difflib 的 SequenceMatcher 来对比两个字符串的差异
    matcher = difflib.SequenceMatcher(None, texta, textb)
    result = []

    # 遍历匹配结果
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'insert':  # 'insert' 表示 textb 中新增的部分
            result.append(textb[j1:j2])
        elif tag == 'replace':  # 'replace' 表示 textb 替换了 texta 中的部分
            result.append(textb[j1:j2])

    # 将新增的 token 拼接成字符串返回
    return ''.join(result)

def uniform_format_of_options(conclustion:str):
    if not isinstance(conclustion, str):
        conclustion = str(conclustion)
        # print(f"conclustion={conclustion}")
    # 使用正则表达式找到所有大写字母
    uppercase_letters = re.findall(r'[A-Z]', conclustion)
    # 利用字典去重并保持顺序
    unique_letters = list(dict.fromkeys(uppercase_letters))
    unique_letters.sort()
    result = json.dumps(unique_letters, ensure_ascii=False)
    return result

def calculate_accuracy_by_two_classify(data: List[Dict]) -> tuple:
    total = len(data)
    exact_match = 0
    partial_match = 0

    for _, item in enumerate(data):
        # 获取ground truth
        ground_truth = eval(item['ground_truth'])
        if "Z" in ground_truth:
            ground_truth = ["否"]
        else:
            ground_truth = ["是"]

        try:
            def replace_final_content(text):
                # 匹配模式: "最终"开头 + 任意字符 + "：" + 任意非空白字符
                pattern = r'最终[^：]*：(\S+)'

                def replace_func(match):
                    # 获取冒号后面的内容
                    content = match.group(1)
                    # 去掉字符串中的非汉字内容
                    content = ''.join(char for char in content if '\u4e00' <= char <= '\u9fff')
                    # 替换为\box{content}格式
                    return f'\\box{{{content}}}'

                # 使用re.sub进行替换
                result = re.sub(pattern, replace_func, text)
                return result

            result_text_old = item['result']
            if "\\boxed{" in result_text_old or "\\box{" in result_text_old:
                # print("has boxed")
                result_text = result_text_old.replace("\\boxed{", "\\box{")
            elif re.search(r"最终\w+：\S+", result_text_old):
                result_text = replace_final_content(result_text_old)
            else:
                result_text = result_text_old

            # 提取result中的json字符串
            if "\\box{" in result_text:
                result = validate_and_extract_box_content(result_text)
                json_str = f'''["{result}"]'''
                predicted_list = json.loads(json_str)
            else:
                print(f"未找到 \\box{{xxx}}包含的内容, 原始result={result_text}")
                continue
            # 计算完全匹配
            if set(predicted_list) == set(ground_truth):
                exact_match += 1

            # 计算部分匹配
            if len(set(predicted_list).intersection(set(ground_truth))) > 0:
                partial_match += 1

        except Exception as e:
            print(str(e))
        predicted_list = None
        ground_truth = None

    exact_match_rate = exact_match / total
    partial_match_rate = partial_match / total
    print(f"完全一致率 = {exact_match}/{total} = {exact_match_rate:.2%}")
    print(f"部分一致率 = {exact_match}/{total} = {partial_match_rate:.2%}")
    return exact_match_rate, partial_match_rate


def trans_result_to_list_to_compute_accuracy(result_text):
    def replace_final_content(text):
        # 匹配模式: "最终"开头 + 任意字符 + "：" + 任意非空白字符
        pattern = r'最终[^：]*：(\S+)'

        def replace_func(match):
            # 获取冒号后面的内容
            content = match.group(1)
            # 去掉字符串中的非汉字内容
            content = ''.join(char for char in content if '\u4e00' <= char <= '\u9fff')
            # 替换为\box{content}格式
            return f'\\box{{{content}}}'

        # 使用re.sub进行替换
        result = re.sub(pattern, replace_func, text)
        return result

    result_text_old = result_text
    if "\\boxed{" in result_text_old or "\\box{" in result_text_old:
        # print("has boxed")
        result_text = result_text_old.replace("\\boxed{", "\\box{")
    elif re.search(r"最终\w+：\S+", result_text_old):
        result_text = replace_final_content(result_text_old)
    else:
        result_text = result_text_old

    # 提取result中的json字符串
    if "\\box{" in result_text:
        result = validate_and_extract_box_content(result_text)
        result = uniform_format_of_options(result)
        predicted_list = json.loads(result)
    else:
        raise Exception(f"未找到 \\box{{xxx}}包含的内容, 原始result={result_text}")

    return predicted_list

def calculate_accuracy_by_multi_classify(data: List[Dict]) -> tuple:
    total = len(data)
    exact_match = 0
    partial_match = 0

    for _, item in enumerate(data):
        # 获取ground truth
        ground_truth = eval(item['ground_truth'])

        try:
            predicted_list = trans_result_to_list_to_compute_accuracy(item['result'])

            # 计算完全匹配
            if set(predicted_list) == set(ground_truth):
                exact_match += 1

            # 计算部分匹配
            if len(set(predicted_list).intersection(set(ground_truth))) > 0:
                partial_match += 1

        except Exception as e:
            print(str(e))
        predicted_list = None
        ground_truth = None

    exact_match_rate = exact_match / total
    partial_match_rate = partial_match / total
    print(f"完全一致率 = {exact_match}/{total} = {exact_match_rate:.2%}")
    print(f"部分一致率 = {exact_match}/{total} = {partial_match_rate:.2%}")
    return exact_match_rate, partial_match_rate


def trans_json_result_to_list_to_compute_accuracy(text):
    # text = extract_json(text)
    # 先用 ast.literal_eval 解析字符串
    parsed_text = ast.literal_eval(text)
    if not isinstance(parsed_text, dict):
    # 然后用 json.loads 转换成 JSON 对象
        json_obj = json.loads(parsed_text)
    else:
        json_obj = parsed_text
    conclustion = json_obj['结论']
    predicted_list_str = uniform_format_of_options(conclustion)
    predicted_list = json.loads(predicted_list_str)
    return predicted_list


def calculate_accuracy_by_multi_classify_on_three_json(data: List[Dict], ground_truth_name="ground_truth", predict_name="result") -> tuple:
    total = len(data)
    exact_match = 0
    partial_match = 0

    for _, item in enumerate(data):
        # 获取ground truth
        ground_truth = eval(item[ground_truth_name])

        try:
            predicted_list = trans_json_result_to_list_to_compute_accuracy(item[predict_name])
            # 计算完全匹配
            if set(predicted_list) == set(ground_truth):
                exact_match += 1

            # 计算部分匹配
            if len(set(predicted_list).intersection(set(ground_truth))) > 0:
                partial_match += 1

        except Exception as e:
            print(str(e))
        predicted_list = None
        ground_truth = None

    exact_match_rate = exact_match / total
    partial_match_rate = partial_match / total
    print(f"完全一致率 = {exact_match}/{total} = {exact_match_rate:.2%}")
    print(f"部分一致率 = {exact_match}/{total} = {partial_match_rate:.2%}")
    return exact_match_rate, partial_match_rate


def encode_image_to_base64(image):
    # 如果图片是RGBA格式，转换为RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # 获取原始图片格式，如果无法获取则默认使用PNG
    format_type = getattr(image, 'format', 'PNG') or 'PNG'

    buffered = io.BytesIO()
    image.save(buffered, format=format_type)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str, format_type.lower()

def concat_base64_image_url(image):
    base64_image, img_format = encode_image_to_base64(image)
    # 根据实际格式构建data URI
    data_uri = f"data:image/{img_format};base64,{base64_image}"
    return data_uri

def trans_multi_classify_to_two_classify(data_list:list):
    result_list = []
    for item in data_list:
        item_result = item['result']
        item_result_box = validate_and_extract_box_content(item_result)
        item_result_box = uniform_format_of_options(item_result_box)
        if "z" in item_result_box.lower():
            item['result'] = "\\box{否}"
        else:
            item['result'] = "\\box{是}"
        result_list.append(item)
    return result_list


def compare_diff_dict_from_two_list(list1, list2):
    # 创建字典来存储每个id对应的处理后的result
    dict1 = {item['id']: validate_and_extract_box_content(item['result']) for item in list1}
    dict2 = {item['id']: validate_and_extract_box_content(item['result']) for item in list2}

    # 找出共同的id
    common_ids = set(dict1.keys()) & set(dict2.keys())

    # 找出result不同的条目
    different_results = []
    for id in common_ids:
        if dict1[id] != dict2[id]:
            # 从原始列表中找到对应的完整记录
            item1 = next(item for item in list1 if item['id'] == id)
            item2 = next(item for item in list2 if item['id'] == id)
            different_results.append({
                'id': id,
                'result1': item1['result'],
                'result2': item2['result']
            })

    return different_results


def process_lists(lista, listb):
    intersection = set()
    for text in lista:
        if text in listb:
            intersection.add(text)

    # 从两个列表中删除交集中的元素
    lista_result = [x for x in lista if x not in intersection]
    listb_result = [x for x in listb if x not in intersection]

    return lista_result, listb_result

def extract_json(s):
    i = s.index('{')
    count = 1  # 当前所在嵌套深度，即还没闭合的'{'个数
    for j, c in enumerate(s[i + 1:], start=i + 1):
        if c == '}':
            count -= 1
        elif c == '{':
            count += 1
        if count == 0:
            break
    assert (count == 0)  # 检查是否找到最后一个'}'
    json_str = s[i:j + 1]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str


def extract_list(text):
    # 清理文本：删除多余的空白字符
    text = text.strip()
    if "平台限流" in text:
        raise Exception(f"current text={text} has '平台限流' limit word.")

    # 如果文本以```json开头
    if text.startswith('```json'):
        # 移除```json和```
        text = text.removeprefix('```json').removesuffix('```').strip()
        try:
            # 尝试直接用eval解析JSON数组
            # （注意：在生产环境中应该使用json.loads，这里用eval是为了简化处理）
            result = eval(text)
            if isinstance(result, list):
                return [str(item).strip() for item in result if item]
        except:
            pass

    # 如果不是JSON格式或JSON解析失败，则使用正则表达式
    patterns = [
        # JSON数组格式
        r'"([^"\\]*(?:\\.[^"\\]*)*)"',  # 匹配JSON中的双引号字符串

        # \box{...}格式
        r'\\box\{([^}]+)\}',  # 匹配\box{}中的内容

        # 其他可能的格式
        r'\'([^\'\\]*(?:\\.[^\'\\]*)*)\'',  # 单引号字符串
    ]

    result = []

    # 尝试所有模式
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            result.extend([match.strip() for match in matches if match])

    # 如果还是没有结果，尝试其他分割方式
    if not result:
        if ';' in text:
            result = [item.strip() for item in text.split(';')]
        elif ',' in text:
            result = [item.strip() for item in text.split(',')]
        else:
            result = [item.strip() for item in text.split('\n')]

    # 清理结果
    result = [s for s in result if s]  # 删除空字符串
    result = [s.strip('[]{}()"\' \t\n') for s in result]  # 删除首尾特殊字符
    result = [re.sub(r'\s+', ' ', s).strip() for s in result]  # 规范化空白字符
    result = list(dict.fromkeys(result))  # 删除重复项

    return result

def validate_and_extract_json(text):
    # 尝试提取JSON字符串
    json_str = extract_json(text)
    # 解析JSON
    json_obj = json.loads(json_str)

    # 验证是否包含必要的字段
    if "conclusion" in json_obj:
        # 拼接内容
        return json_obj['conclusion']

def validate_and_extract_box_content(text):
    if "平台限流" in text:
        raise Exception(f"current text={text} has '平台限流' limit word.")
    if "\\boxed{" in text:
        text = text.replace("\\boxed{", "\\box{")
    if "\\text{" in text:
        text = text.replace("\\text{", "\\box{")
    pattern = r'\\box{(.*?)}'
    match = re.search(pattern, text)
    result = ""
    if match:
        content = match.group(1)
        # print(content)
        result += content
    else:
        return None
    return result.replace("{","").replace("}","")


def validate_and_extract_three_json(text):
    if "平台限流" in text:
        raise Exception(f"current text={text} has '平台限流' limit word.")

    # 尝试提取JSON字符串
    json_str = extract_json(text)
    # 解析JSON
    json_obj = json.loads(json_str)

    # 验证是否包含必要的字段
    if "分析" in json_obj and "关键词" in json_obj and "结论" in json_obj:
        return json.dumps(json_obj, ensure_ascii=False, indent=2)
    raise Exception(f"current text={text} hasn't three json format.")


def call_idealab_api(prompt, image_url:str, model_name:str):
    if isinstance(prompt, str):
        if image_url:
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        }
                    }
                ]
            }]
        else:
            messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    client = OpenAI(
        api_key=os.getenv("api_key", ""),
        base_url=os.getenv("base_url", ""),
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=8192,
        temperature=0.8,
        stream=True
    )
    result = ""

    for chunk in completion:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            result += content
    return result, None

def call_idealab_api_without_stream(prompt, image_url:str, model_name:str):
    if isinstance(prompt, str):
        if image_url:
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        }
                    }
                ]
            }]
        else:
            messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    client = OpenAI(
        api_key=os.getenv("api_key", ""),
        base_url=os.getenv("base_url", ""),
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=8192,
        temperature=0.8,
        stream=False
    )
    result = completion.choices[0].message.content
    print(f"result={result}")
    return result, None

def call_qwen3_api(prompt, model_name="qwen3", enable_thinking=True):
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    client = OpenAI(
        api_key=os.getenv("api_key", ""),
        base_url=os.getenv("base_url", ""),
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=20480,
        temperature=0.8,
        stream=True,
        extra_body={"enable_thinking": enable_thinking}
    )
    result = ""
    reasoning_content = ""

    if not enable_thinking:
        for chunk in completion:
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
    else:
        for chunk in completion:
            if hasattr(chunk.choices[0].delta, "reasoning_content"):
                reasoning_content += chunk.choices[0].delta.reasoning_content
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
    return result,reasoning_content


def messages_builder_example(prompt,url,few_shot=False):
    content = [
        {
            "type": "text",
            "text": prompt
        },
        {
            "type": "image_url",
            "image_url": {
                "url": url,
                "detail": "high"
            }
        }
    ]
    messages = [{"role": "user", "content": content}]
    if few_shot:
        messages = {"role": "user", "content": content}
    return messages


def get_all_file_name_in_special_folder(target_dir:str, is_need_suffix=False):
    file_list = [
        os.path.splitext(f)[0] if not is_need_suffix else f  # 分割文件名和后缀，取前半部分
        for f in os.listdir(target_dir)  # 遍历目录下的文件名
        if os.path.isfile(os.path.join(target_dir, f))  # 拼接完整路径并验证是文件
        and not f.startswith(".")  # 排除隐藏文件

    ]
    return file_list


def get_inference_result_and_check_accuracy(data_list:list):
    full_correnct_match = 0
    part_correnct_match = 0
    total_count = 0
    for index, item in enumerate(data_list):
        try:
            ground_truth_list = eval(item['ground_truth'])
            if "\\box" in item['generate_results']:
                box_content = validate_and_extract_box_content(item['generate_results'])
                box_content_str = uniform_format_of_options(box_content)
                predict_list = json.loads(box_content_str)
            elif validate_and_extract_three_json(item['generate_results']):
                predict_json = validate_and_extract_three_json(item['generate_results'])
                predict_list_str = json.loads(predict_json)['结论']
                predict_list_str = uniform_format_of_options(predict_list_str)
                predict_list = json.loads(predict_list_str)
            elif item['generate_results'].startswith('[') and item['generate_results'].endswith(']'):
                predict_list_str = uniform_format_of_options(item['generate_results'])
                predict_list = json.loads(predict_list_str)
            elif "ox{" in item['generate_results']:
                item['generate_results'] = item['generate_results'].replace("ox{", "\\box{")
                box_content = validate_and_extract_box_content(item['generate_results'])
                box_content_str = uniform_format_of_options(box_content)
                predict_list = json.loads(box_content_str)
            else:
                print(f"current item['generate_results'] not a correct format = {json.dumps(item['generate_results'], ensure_ascii=False)}")
                continue
            full_correnct = set(ground_truth_list) == set(predict_list)
            if full_correnct:
                full_correnct_match += 1
            if len(set(ground_truth_list).intersection(set(predict_list))) > 0:
                part_correnct_match += 1
            total_count += 1

        except Exception as e:
            print(f"error")
            continue
    print(f"format error count = {len(data_list) - total_count}")
    print(f"full accuracy = {full_correnct_match}/{total_count} = {full_correnct_match / total_count:.4f}")
    print(f"part accuracy = {part_correnct_match}/{total_count} = {part_correnct_match / total_count:.4f}")


def split_rag_dataset(input_list: list, document_ratio=0.2) -> tuple:
    """
    将输入的列表随机划分为document集和query集

    Args:
        input_list (list): 包含字符串的输入列表
        document_ratio (float): document集的比例，默认0.2

    Returns:
        tuple: (document_list, query_list)
    """
    # 检查输入参数
    if not 0 < document_ratio < 1:
        raise ValueError("document_ratio must be between 0 and 1")

    # 复制列表并随机打乱
    shuffled_list = input_list.copy()
    random.shuffle(shuffled_list)

    # 计算划分点
    split_point = int(len(shuffled_list) * document_ratio)

    # 划分列表
    document_list = shuffled_list[:split_point]
    query_list = shuffled_list[split_point:]

    return document_list, query_list


def retrieve_similar_document(document_list: list, query: str) -> str:
    """
    在document_list中找到与query最相似的文档

    Args:
        document_list (list): 文档列表
        query (str): 查询文本

    Returns:
        str: 最相似的文档
    """
    # 如果文档列表为空，直接返回None
    if not document_list:
        return None

    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer()

    # 将所有文档和查询文本组合在一起进行向量化
    all_texts = document_list + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # 获取查询文本的向量（最后一个）
    query_vector = tfidf_matrix[-1]

    # 计算查询文本与所有文档的相似度
    similarities = cosine_similarity(query_vector, tfidf_matrix[:-1])[0]

    # 找到最相似的文档的索引
    most_similar_idx = np.argmax(similarities)

    return document_list[most_similar_idx]


def messages_builder_example_one_shot_text(one_shot_question:str, one_shot_answer:str, reasoning_question:str):
    dialog_message = []
    dialog_message.append({"role": "user", "content": one_shot_question})
    dialog_message.append({"role": "assistant", "content": one_shot_answer})
    dialog_message.append({"role": "user", "content": reasoning_question})
    return dialog_message

def is_limit_api(response:str):
    if "conflict" in response or "限流" in response or "TOO_MANY_REQUESTS" in response or "error" in response:
        raise Exception(f"current text={response} has '限流' limit word.")
    return False

def compute_accuracy_by_filename(file_name:str, content_type_list:list):
    with open(file_name, 'r', encoding="utf-8") as f:
        result_list = json.load(f)

    datas = []
    if content_type_list and len(content_type_list) > 0:
        for item in result_list:
            if item['content_type'] in content_type_list:
                datas.append(item)
    else:
        datas = result_list
    get_inference_result_and_check_accuracy(datas)

def extract_complex_rules_from_prompt(prompt:str):
    """
    截取"# 管控类型"和"# 输出格式"中间的所有字符串内容
    参数:
        text: 输入的字符串
    返回:
        截取的内容，如果找不到标记则返回自身
    """
    start_marker = "# 管控类型"
    end_marker = "# 输出格式"

    start_index = prompt.find(start_marker)
    end_index = prompt.find(end_marker)

    # 如果两个标记都存在且顺序正确
    if start_index != -1 and end_index != -1 and start_index < end_index:
        # 从start_marker之后开始截取，到end_marker之前结束
        start_pos = start_index + len(start_marker)
        content = prompt[start_pos:end_index]
        return content.strip()  # 去除首尾空白字符

    return prompt