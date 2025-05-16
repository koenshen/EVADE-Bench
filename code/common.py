import time
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import concurrent.futures
import common_io
from typing import ByteString
import json, re, random
import copy
from openai import OpenAI


def _download(slice_count, slice_id, tableName, selected_cols):
    reader = common_io.table.TableReader(tableName,
                                         selected_cols=selected_cols,
                                         slice_id=slice_id,
                                         slice_count=slice_count)
    total = reader.get_row_count()
    if selected_cols == "":
        selected_cols = [x[0] for x in reader.get_schema()]
    else:
        selected_cols = selected_cols.split(',')

    data = reader.read(num_records=total, allow_smaller_final_batch=True)
    df = {}
    for i, c in enumerate(selected_cols):
        df[c] = pd.Series([row[i].decode() if isinstance(row[i], ByteString) else row[i] for row in data])
    del data
    output = pd.DataFrame(df)
    return output


def read_data_from_odps(tableName, selected_cols='', num_workers=10, rank=0, world_size=1):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            future = executor.submit(
                _download,
                num_workers * world_size,
                i + rank * num_workers,
                tableName,
                selected_cols
            )
            futures.append(future)
        data = [f.result() for f in futures]

    return pd.concat(data)


def messages_builder_example(prompt, url, few_shot=False):
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


def messages_builder_example_one_shot_image(one_shot_question: str, one_shot_question_url: str, one_shot_answer: str,
                                            reasoning_question: str, reasoning_question_url: str):
    dialog_message = []
    dialog_message.append(messages_builder_example(one_shot_question, one_shot_question_url, few_shot=True))
    dialog_message.append({"role": "assistant", "content": one_shot_answer})
    dialog_message.append(messages_builder_example(reasoning_question, reasoning_question_url, few_shot=True))
    return dialog_message


def messages_builder_example_one_shot_text(one_shot_question: str, one_shot_answer: str, reasoning_question: str):
    dialog_message = []
    dialog_message.append({"role": "user", "content": one_shot_question})
    dialog_message.append({"role": "assistant", "content": one_shot_answer})
    dialog_message.append({"role": "user", "content": reasoning_question})
    return dialog_message


def write_train_data_to_table(data_list: list, write_to_table_name):
    ### 和直写odps唯一差异
    # write_to_table_name = "odps://alimama_intern_dev/tables/alimama_disease_dpo_v1"
    writer = common_io.table.TableWriter(write_to_table_name, slice_id=0)
    w_result = []
    for index, row in enumerate(data_list):
        w_result.append((row['id'], row['question'], row['content'], row['content_type'], row['分析'], row['关键词'],
                         row['选项'], row['extra']))
        if len(w_result) == 100:
            writer.write(w_result, (0, 1, 2, 3, 4, 5, 6, 7))
            print(f"当前是第{index + 1}条，本次已插入了{len(w_result)}条数据到{write_to_table_name}中.")
            w_result = []

    if len(w_result) > 0:
        writer.write(w_result, (0, 1, 2, 3, 4, 5, 6, 7))
        print(f"还有剩余, 本次已插入了{len(w_result)}条数据到{write_to_table_name}中.")
        w_result = []
    writer.close()


def write_train_data_to_omega_table(old_list: list, write_to_table_name, type="text", few_shot=False):
    data_list = []
    for index, item in enumerate(old_list):
        message = [{"role": "user", "content": item['question']}]
        if few_shot:
            prompt = item['question']
        elif type == 'text':
            prompt = json.dumps(message, ensure_ascii=False, indent=2)
        else:
            prompt = json.dumps(messages_builder_example(item['question'], item['content']), ensure_ascii=False,
                                indent=2)
        item_dict = {
            "id": item['id'],
            "prompt": prompt,
            "result": "",
        }
        data_list.append(item_dict)
    print(f"数据处理完成，数据量为：{len(data_list)}")

    writer = common_io.table.TableWriter(write_to_table_name, slice_id=0)
    w_result = []
    for index, row in enumerate(data_list):
        w_result.append((row['id'], row['prompt'], row['result']))
        if len(w_result) == 100:
            writer.write(w_result, (0, 1, 2))
            print(f"当前是第{index + 1}条，本次已插入了{len(w_result)}条数据到{write_to_table_name}中.")
            w_result = []

    if len(w_result) > 0:
        writer.write(w_result, (0, 1, 2))
        print(f"还有剩余, 本次已插入了{len(w_result)}条数据到{write_to_table_name}中.")
    writer.close()


def write_train_data_to_idealab_table(data_list: list, write_to_table_name, type="text"):
    writer = common_io.table.TableWriter(write_to_table_name, slice_id=0)
    w_result = []
    for index, row in enumerate(data_list):
        if type == "text":
            content = None
        else:
            content = row['content']
        w_result.append((row['id'], row['question'], content))
        if len(w_result) == 100:
            writer.write(w_result, (0, 1, 2))
            print(f"当前是第{index + 1}条，本次已插入了{len(w_result)}条数据到{write_to_table_name}中.")
            w_result = []

    if len(w_result) > 0:
        writer.write(w_result, (0, 1, 2))
        print(f"还有剩余, 本次已插入了{len(w_result)}条数据到{write_to_table_name}中.")
        w_result = []
    writer.close()


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
    return s[i:j + 1]


def get_human_table_datas_to_list(original_table_name: str):
    original_datas = read_data_from_odps(original_table_name)
    original_data_list = []
    for index, item in original_datas.iterrows():
        new_item = {}
        for key, value in item.items():
            new_item[key] = value
        original_data_list.append(new_item)
    random.shuffle(original_data_list)
    return original_data_list


def label_contenttype_by_prompt(item):
    question = item['question']
    if "A. 增高功效直接描述：" in question:
        item['content_type'] = "tall"
    elif "A. 缩阴直接描述：" in question:
        item['content_type'] = 'suoyin'
    elif "A. 特定减肥商品：" in question:
        item['content_type'] = 'jianfei'
    elif "A.甲乙丙类传染病" in question:
        item['content_type'] = 'disease'
    elif "A. 壮阳特定商品：" in question:
        item['content_type'] = 'zhuangyang'
    elif "A. 丰胸功效直接描述：" in question:
        item['content_type'] = 'fengxiong'
    else:
        print(f"have not any match content_type with {question}")
    return item


def call_idealab_and_concat_stream_response_to_whole_str(messages: list, model_name="gpt-4o-0513", enable_thinking=True):
    client = OpenAI(
        api_key="xxx",
        base_url="xxx",
    )
    is_stream = True
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=8192,
        stream=is_stream,
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
    print(f'''content={messages[0]['content'].split("# 给定信息")[1].strip()}, final response: {result}''')
    return result, reasoning_content


def remove_repeate_item(whole_datas: list):
    whole_set = set()
    final_datas = []
    for item in whole_datas:
        if item['content'] in whole_set:
            print(f"重复的content={item['content']}")
            continue
        else:
            whole_set.add(item['content'])
            final_datas.append(item)
    return final_datas


def remove_repeate_item_of_allinone(whole_datas: list):
    whole_set = set()
    final_datas = []
    for item in whole_datas:
        no_repeate_str = item['content'] + item['extra']
        if no_repeate_str in whole_set:
            print(f"重复的content={no_repeate_str}")
            continue
        else:
            whole_set.add(no_repeate_str)
            final_datas.append(item)
    return final_datas


# 将模型输出的回答格式,保持与 human-v1 的回答格式一致
def uniform_format_of_options(conclustion: str):
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


def call_deepseek_vl2(messages: list):
    from openai import OpenAI
    # 免费的key
    client = OpenAI(api_key="xxx", base_url="xxx")

    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-vl2",
        messages=messages,
        stream=False,
        temperature=0.8,
        max_tokens=256,  # 超过256的mantoken,会导致deepseekvl2无法回答疾病问题,因为超出了4k
    )
    final_res = response.choices[0].message.content
    return final_res


# 共用的模块,不用在百炼idealab和omega各自写一份代码
def handle_single_item_and_extract_analysis_and_conclustion(final_data: list, id_answer_dict: dict, new_item: dict,
                                                            id_key: str, error_case: int):
    whole_answer = id_answer_dict[id_key]
    if not whole_answer:
        error_case += 1
        print(f"{error_case}警告：id={id_key} 对应的回答为 None，跳过处理")
        return final_data, error_case

    if len(new_item['content']) < 3:
        print(f"{error_case}警告：id={id_key} 对应的content小于3个字符={new_item['content']}，跳过处理")
        return final_data, error_case

    # 对于含有role=assistant的json,需要提取content的内容
    if isinstance(whole_answer, dict) and "content" in whole_answer.keys():
        whole_answer = whole_answer['content']
    elif isinstance(whole_answer, str) and "content" in whole_answer:
        try:
            whole_answer_json = json.loads(whole_answer)
            whole_answer = whole_answer_json['content']
        except Exception as e:
            print(
                "将whole_answer={whole_answer}转为json再提取content的方法失败. 默认whole_answer直接就是模型的最内层回答了.")

    try:
        extract_json_str = extract_json(whole_answer)
        template_json = json.loads(extract_json_str)
        if '分析' not in template_json.keys() or '关键词' not in template_json.keys() or '结论' not in template_json.keys():
            return final_data, error_case

        new_item['分析'] = template_json['分析']
        new_item['关键词'] = json.dumps(template_json['关键词'], ensure_ascii=False)
        new_item['选项'] = uniform_format_of_options(template_json['结论'])
        # 如果既有其他又有非其他的选项,则不要
        choices_list = json.loads(new_item['选项'])
        if "Z" in choices_list and len(choices_list) > 1:
            print(f"当前item的模型回答中,选项既有Z其他又有非其他的选项={new_item['选项']}")
            return final_data, error_case
        final_data.append(new_item)
    except Exception as e:
        if "'tuple' object has no attribute 'append'" in str(e):
            print("")
        pattern = r'"结论"\s*:\s*"([^"]*)"'
        match = re.search(pattern, whole_answer)
        if match:
            conclusion_value = match.group(1)
            new_item['分析'] = whole_answer
            new_item['关键词'] = ""
            new_item['选项'] = uniform_format_of_options(conclusion_value)
            # 如果既有其他又有非其他的选项,则不要
            choices_list = json.loads(new_item['选项'])
            if "Z" in choices_list and len(choices_list) > 1:
                print(f"当前item的模型回答中,选项既有Z其他又有非其他的选项={new_item['选项']}")
                return final_data, error_case
            final_data.append(new_item)
        else:
            error_case += 1
            print(f"未找到 '结论' 对应的值, 原始回答={whole_answer}, error_case={error_case}")

    return final_data, error_case


# 处理单个的deepseekvl2的结果
def handle_one_item_for_deepseekvl2(new_item: dict, current_index: int, total_len: int):
    utc_now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    bj_time = utc_now.astimezone(ZoneInfo("Asia/Shanghai"))
    result = ""
    image_url = new_item['content']
    prompt = new_item['question'].split("# 给定信息")[0].strip()
    content = [
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
    messages = [{"role": "user", "content": content}]
    for i in range(10):
        print(
            f'''{current_index}/{total_len}条数据开始被处理, 当前时间={bj_time.strftime("%Y-%m-%d %H:%M:%S")}, url={new_item['content']}''')
        try:
            result = call_deepseek_vl2(messages)
            if result and "分析" in result and "关键词" in result and "结论" in result:
                break
        except Exception as call_e:
            print(call_e)
            if "'Image url should be a valid url or should like data" in str(call_e):
                break
            if "Connection error." in str(call_e):
                break
            print(
                f"第{current_index}/{total_len}条数据在第{i + 1}次调用deepseek-vl2遇到失败,输出result={result},需要等待下一次运行")
            time.sleep(5)
            continue
    return result


# 处理单个的qwen3的结果
def handle_one_item_for_qwen3(new_item: dict, current_index: int, total_len: int, model_name: str,
                              enable_thinking: bool):
    utc_now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    bj_time = utc_now.astimezone(ZoneInfo("Asia/Shanghai"))
    messages = [{"role": "user", "content": new_item['question']}]
    result = ""
    reasoning_content = ""
    for i in range(100):
        print(
            f'''{current_index}/{total_len}条数据开始被处理, 当前时间={bj_time.strftime("%Y-%m-%d %H:%M:%S")}, content={new_item['content']}''')
        try:
            result, reasoning_content = call_idealab_and_concat_stream_response_to_whole_str(messages,
                                                                                             model_name=model_name,
                                                                                             enable_thinking=enable_thinking)
            if result and "分析" in result and "关键词" in result and "结论" in result:
                break
        except Exception as call_e:
            print(call_e)
            if "Output data may contain inappropriate content" in str(call_e):
                break
            print(
                f"第{current_index}/{total_len}条数据在第{i + 1}次调用遇到失败,输出result={result},需要等待3s后的下一次运行")
            time.sleep(3)
            continue
    return result, reasoning_content


# idealab 调用VL模型处理的结果
def handle_one_item_for_idealab_vl(item_data: dict, current_index: int, total_len: int, model_name='gpt-4o-0806',
                                   is_rag_format=False) -> str:
    def messages_builder_example(question: str, image_url: str = None) -> dict:
        """
        Builds a single user message dictionary with text and an optional image.
        """
        content_payload = [{"type": "text", "text": question}]
        if image_url:
            content_payload.append({"type": "image_url", "image_url": {"url": image_url}})
        return {"role": "user", "content": content_payload}

    def messages_builder_example_one_shot_image(
            one_shot_question: str,
            one_shot_question_url: str,
            one_shot_answer: str,
            reasoning_question: str,
            reasoning_question_url: str
    ) -> list:
        """
        Builds the message list for a one-shot image-based query.
        """
        dialog_message = []
        # Example turn
        dialog_message.append(messages_builder_example(one_shot_question, one_shot_question_url))
        dialog_message.append({"role": "assistant", "content": one_shot_answer})
        # Actual query turn
        dialog_message.append(messages_builder_example(reasoning_question, reasoning_question_url))
        return dialog_message

    def _simple_call_idealab_vl(messages: list, model_name: str, enable_thinking: bool = False) -> str:
        full_response_content = ""
        if not idealab_client:
            # This check is now more prominent before the actual call attempt
            # The caller function handle_one_item_for_idealab_vl also has a check,
            # but this ensures _simple_call_idealab_vl itself is safe.
            print(f"    错误 (内部调用): IdeaLab OpenAI客户端未初始化。无法调用IdeaLab VL。")
            return ""  # Return empty string or raise an error, depending on desired handling

        completion = idealab_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=8192,  # As per original code
            stream=True,
            # extra_body={"enable_thinking": enable_thinking}
        )
        for chunk in completion:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_response_content += delta.content
        return full_response_content

    IDEALAB_VL_MODEL_NAME = model_name
    IDEALAB_API_KEY = "xxx"
    IDEALAB_BASE_URL = "xxx"

    idealab_client = None
    if IDEALAB_API_KEY:
        try:
            idealab_client = OpenAI(api_key=IDEALAB_API_KEY, base_url=IDEALAB_BASE_URL, )
        except Exception as e:
            print(f"初始化IdeaLab OpenAI客户端失败: {e}")
    else:  # pragma: no cover
        print("IdeaLab OpenAI客户端未初始化，因为IDEALAB_API_KEY未提供。")

    bj_time_str = "N/A"
    try:
        utc_now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        bj_time = utc_now.astimezone(ZoneInfo("Asia/Shanghai"))
        bj_time_str = bj_time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as tz_e:  # pragma: no cover
        bj_time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S (UTC fallback)")
        print(f"获取北京时间失败 ({tz_e})，使用UTC时间。")

    result = ""
    if not is_rag_format:
        # Extract data for one-shot and reasoning
        one_shot_question = item_data.get('one_shot_question', '')
        one_shot_question_url = item_data.get('one_shot_question_url',
                                              '')  # Can be None or empty if example is text-only
        one_shot_answer = item_data.get('one_shot_answer', '')
        reasoning_question_raw = item_data.get('reasoning_question', '')  # This is the main question for the new image
        reasoning_question_url = item_data.get('reasoning_question_url', '')  # This is the main image URL

        if not reasoning_question_url or not reasoning_question_raw:
            print(f"警告: {current_index}/{total_len} 数据中推理图片URL或问题为空，跳过。")
            return ""

        # Process the reasoning question to extract the prompt part (same logic as before)
        reasoning_prompt_text = reasoning_question_raw.split("# 给定信息")[0].strip()

        # Construct messages using the one-shot builder
        messages = messages_builder_example_one_shot_image(
            one_shot_question=one_shot_question,
            one_shot_question_url=one_shot_question_url,
            one_shot_answer=one_shot_answer,
            reasoning_question=reasoning_prompt_text,  # Pass the processed prompt
            reasoning_question_url=reasoning_question_url
        )
        log_image_url = reasoning_question_url
    else:
        _question = item_data.get('question', [])
        question = json.loads(_question)

        messages = []
        if isinstance(question[0], list):
            messages.append(question[0][0])
        else:
            messages.append(question[0])
        if isinstance(question[1], list):
            messages.append(question[1][0])
        else:
            messages.append(question[1])
        if isinstance(question[2], list):
            messages.append(question[2][0])

        log_image_url = item_data.get('content', '')
        # print(messages)
        # returnclear

    for i in range(10):  # Max 10 retries
        print(f'''{current_index}/{total_len}条数据开始被IdeaLab处理, 时间={bj_time_str}, url={log_image_url}''')
        try:
            result = _simple_call_idealab_vl(messages, IDEALAB_VL_MODEL_NAME, enable_thinking=False)
            if result and "分析" in result and "关键词" in result and "结论" in result:
                break
        except Exception as call_e:
            error_str = str(call_e)
            print(f"    错误 (IdeaLab call): {error_str}")

            print(f"    第{current_index}/{total_len}条数据在第{i + 1}次调用IdeaLab VL失败, result='{result}'")
            if i < 9:  # If not the last attempt
                time.sleep(5)  # Wait before retrying
            else:
                print(f"    达到最大重试次数，项目 {current_index}/{total_len} 处理失败。")
    return result


# 用deepseekvl2处理单一风险的方法
def handle_odps_image_and_call_deepseekvl2_api(risk_type: str, original_datas: list, new_table_name: str, prompt: str,
                                               already_has_local_file=""):
    final_data = []
    already_local_dict = {}
    error_case = 0
    temp_save_local_file_name = f"{risk_type}-deepseekvl2.json"

    if already_has_local_file:
        with open(already_has_local_file, 'r', encoding="utf-8") as f:
            already_local_json = json.load(f)
        already_local_dict = {}
        for index, item in enumerate(already_local_json):
            already_local_dict[item['id']] = item
        temp_save_local_file_name = already_has_local_file.split(".json")[0] + "_v1.json"

    for index, item in enumerate(original_datas):
        new_item = {}
        for key, value in item.items():
            new_item[key] = value

        if already_has_local_file and item['id'] in already_local_dict.keys():
            print(f'''{index}/{len(original_datas)}条数据已经被处理过,无需调用api,直接从本地文件获取完整的item.''')
            new_item = already_local_dict[item['id']]
            final_data.append(new_item)
            continue

        result = handle_one_item_for_deepseekvl2(new_item=new_item, current_index=index, total_len=len(original_datas))

        id_answer_dict = {}
        id_answer_dict[str(new_item['id'])] = result
        final_data, error_case = handle_single_item_and_extract_analysis_and_conclustion(final_data=final_data,
                                                                                         id_answer_dict=id_answer_dict,
                                                                                         new_item=new_item,
                                                                                         id_key=str(new_item['id']),
                                                                                         error_case=error_case)

        # 暂时保存到本地的json文件,防止中途报错需要重新开始
        with open(temp_save_local_file_name, 'w', encoding="utf-8") as f:
            f.write(json.dumps(final_data, ensure_ascii=False, indent=2))

    write_train_data_to_table(final_data, new_table_name)


# 用deepseekvl2处理所有风险的方法
def handle_all_risk_image_and_call_deepseekvl2_api(original_datas: list, new_table_name: str,
                                                   already_has_local_file=""):
    final_data = []
    already_local_dict = {}
    error_case = 0
    temp_save_local_file_name = f"all_risk_image-deepseekvl2-len{len(original_datas)}.json"

    if already_has_local_file:
        with open(already_has_local_file, 'r', encoding="utf-8") as f:
            already_local_json = json.load(f)
        already_local_dict = {}
        for index, item in enumerate(already_local_json):
            already_local_dict[item['id']] = item
        temp_save_local_file_name = already_has_local_file.split(".json")[0] + "_v1.json"

    for index, item in enumerate(original_datas):
        new_item = {}
        for key, value in item.items():
            new_item[key] = value

        if already_has_local_file and new_item['id'] in already_local_dict.keys():
            print(f'''{index}/{len(original_datas)}条数据已经被处理过,无需调用api,直接从本地文件获取完整的item.''')
            new_item = already_local_dict[new_item['id']]
            final_data.append(new_item)
            continue

        result = handle_one_item_for_deepseekvl2(new_item=new_item, current_index=index, total_len=len(original_datas))

        id_answer_dict = {}
        id_answer_dict[str(new_item['id'])] = result
        final_data, error_case = handle_single_item_and_extract_analysis_and_conclustion(final_data=final_data,
                                                                                         id_answer_dict=id_answer_dict,
                                                                                         new_item=new_item,
                                                                                         id_key=str(new_item['id']),
                                                                                         error_case=error_case)

        # 暂时保存到本地的json文件,防止中途报错需要重新开始
        with open(temp_save_local_file_name, 'w', encoding="utf-8") as f:
            f.write(json.dumps(final_data, ensure_ascii=False, indent=2))

    write_train_data_to_table(final_data, new_table_name)


def read_bailian_jsonl_and_concat_final_data(original_table_name, write_table_name, data_file_path,
                                             use_old_extra_id=False, risk_type=""):
    bailian_result = {}
    final_data = []
    error_case = 0
    with open(data_file_path, "r", encoding="utf-8") as file:
        for line in file:
            json_obj = json.loads(line)
            bailian_result[json_obj['custom_id']] = json_obj['response']['body']['choices'][0]['message']

    # 可以自己选择传入的是原始table名或者传入一个list,如果传入list可以节省很大一部分的读表时间
    original_list = []
    if isinstance(original_table_name, str):
        original_datas = read_data_from_odps(original_table_name)
        # 组装原始数据和回答
        for index, item in original_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value
            original_list.append(new_item)

    if isinstance(original_table_name, list):
        original_list = copy.deepcopy(original_table_name)

    for index, new_item in enumerate(original_list):
        if risk_type and (new_item['content_type'] != risk_type):
            continue
        id = str(new_item['id'])
        if use_old_extra_id:
            id = new_item['extra']
        if id not in bailian_result.keys():
            print(f"can not find the {id} in the bailian results.")
            continue
        final_data, error_case = handle_single_item_and_extract_analysis_and_conclustion(final_data=final_data,
                                                                                         id_answer_dict=bailian_result,
                                                                                         new_item=new_item, id_key=id,
                                                                                         error_case=error_case)

    random.shuffle(final_data)
    write_train_data_to_table(final_data, write_table_name)


# use_od_extra_id的意义在于,如果这条数据当初是分风险跑的那么idealab的output就只有旧id,那么就得替换掉新id
# risk_type的意义在于,每次都是先找human的数据再根据human匹配模型的结果,但这样会导致有很多human的数据找不到对应结果(在human数据明显大于idealab的情况下,比如human有6风险,而idealab只是单个单个风险的跑)
def read_idealab_and_concat_final_data(original_table_name, write_table_name, idealab_table_name, model_name,
                                       use_old_extra_id=False, risk_type=""):
    idealab_result = {}
    idealab_data_df = read_data_from_odps(idealab_table_name)
    for _, item in idealab_data_df.iterrows():
        # item_dict = {
        #     "id": item[0],
        #     "question": item[1],
        #     "answer": item[2],
        #     "finish_time": item[3],
        #     "success": item[4],
        #     "model": item[9],
        # }
        if str(item[9]) == model_name:
            idealab_result[item[0]] = item[2]

    hasnot_key_count = 0
    final_data = []
    error_case = 0

    # 可以自己选择传入的是原始table名或者传入一个list,如果传入list可以节省很大一部分的读表时间
    original_list = []
    if isinstance(original_table_name, str):
        original_datas = read_data_from_odps(original_table_name)
        # 组装原始数据和回答
        for index, item in original_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value
            original_list.append(new_item)

    if isinstance(original_table_name, list):
        original_list = copy.deepcopy(original_table_name)

    for index, new_item in enumerate(original_list):
        if risk_type and (new_item['content_type'] != risk_type):
            continue
        id = str(new_item['id'])
        if use_old_extra_id:
            id = new_item['extra']
        if id not in idealab_result.keys():
            hasnot_key_count += 1
            print(f"NO.{hasnot_key_count}, can not find the {id} in the idealab results.")
            continue
        final_data, error_case = handle_single_item_and_extract_analysis_and_conclustion(final_data=final_data,
                                                                                         id_answer_dict=idealab_result,
                                                                                         new_item=new_item, id_key=id,
                                                                                         error_case=error_case)

    random.shuffle(final_data)
    write_train_data_to_table(final_data, write_table_name)


def read_omega_and_concat_final_data(original_table_name, write_table_name, omega_output_table_name,
                                     use_old_extra_id=False, risk_type=""):
    final_data = []
    error_case = 0
    omega_output_datas = read_data_from_odps(omega_output_table_name)
    id_answer_dict = {}
    for index, row in omega_output_datas.iterrows():
        if not row['result']:
            continue
        id_answer_dict[row['id']] = row['result']

    # 可以自己选择传入的是原始table名或者传入一个list,如果传入list可以节省很大一部分的读表时间
    original_list = []
    if isinstance(original_table_name, str):
        original_datas = read_data_from_odps(original_table_name)
        # 组装原始数据和回答
        for index, item in original_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value
            original_list.append(new_item)

    if isinstance(original_table_name, list):
        original_list = copy.deepcopy(original_table_name)

    for index, new_item in enumerate(original_list):
        if risk_type and (new_item['content_type'] != risk_type):
            continue
        id = str(new_item['id'])
        if use_old_extra_id:
            id = new_item['extra']
        if id not in id_answer_dict.keys():
            print(f"can not find the {id} in the omega results.")
            continue
        final_data, error_case = handle_single_item_and_extract_analysis_and_conclustion(final_data=final_data,
                                                                                         id_answer_dict=id_answer_dict,
                                                                                         new_item=new_item, id_key=id,
                                                                                         error_case=error_case)

    random.shuffle(final_data)
    write_train_data_to_table(final_data, write_table_name)


def check_concathumandata_is_consistent_with_omegainput(human_datas: list, omega_datas: list):
    omega_dict = {}
    for index, item in enumerate(omega_datas):
        omega_dict[str(item['id'])] = item['prompt']

    final_datas = []
    for index, item in enumerate(human_datas):
        if str(item['id']) in omega_dict.keys():
            omega_prompt = omega_dict[str(item['id'])]
            if item['content'] not in omega_prompt:
                print(f"id={str(item['id'])}相同但是content不一致={item['content']}")
            else:
                final_datas.append(item)
    return final_datas


def read_mdl_log_and_concat_data(original_table_name, write_table_name, log_file_path):
    log_responses = {}

    with open(log_file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("NO."):
                try:
                    # 提取 id
                    id_match = re.search(r"id=(\d+)", line)
                    if not id_match:
                        print(f"未找到 id: {line.strip()}\n")
                        continue
                    record_id = id_match.group(1)

                    # 提取 response
                    cleaned_line = line.replace("\\n", "").replace("\\t", "").replace('\\"', '"')
                    response_match = re.search(r'response\s*=\s*"```json([\s\S]*?)```', cleaned_line)
                    if not response_match:
                        print(f"未找到 response: {cleaned_line}\n")
                        continue
                    response_json_str = response_match.group(1).strip()
                    try:
                        parsed_data = json.loads(response_json_str)
                        if '分析' not in parsed_data or '关键词' not in parsed_data or '结论' not in parsed_data:
                            print(f"警告：record_id={record_id} 缺少必要字段，跳过该条数据")
                            continue
                        log_responses[record_id] = parsed_data
                    except json.JSONDecodeError:
                        pattern = r'"结论"\s*:\s*"([^"]*)"'
                        match = re.search(pattern, response_json_str)
                        if match:
                            conclusion_value = match.group(1)
                            log_responses[record_id] = {
                                "分析": response_json_str,
                                "关键词": [],
                                "结论": conclusion_value
                            }
                        else:
                            print(f"json格式错误且未找到 '结论'\n: {line.strip()}")
                            continue
                except Exception as e:
                    print(f"解析日志行失败: {line.strip()}, 错误: {e}")

    print(f"读取日志文件成功, 数据量: {len(log_responses)}")

    # 读取原始数据表
    original_datas = read_data_from_odps(original_table_name)
    print(f"读取原始数据表成功, 数据量: {len(original_datas)}")

    # 组装最终数据
    final_data = []
    for index, item in original_datas.iterrows():
        record_id = str(item['id'])
        if record_id not in log_responses:
            continue  # 如果日志中没有对应的 id，跳过

        new_item = {}
        for key, value in item.items():
            new_item[key] = value

        response_data = log_responses[record_id]

        new_item['分析'] = response_data['分析']
        new_item['关键词'] = json.dumps(response_data['关键词'], ensure_ascii=False)
        new_item['选项'] = uniform_format_of_options(response_data['结论'])
        final_data.append(new_item)

    random.shuffle(final_data)
    # 随机查看1条数据
    for i in range(1):
        print(f"final_data[{i}]: {final_data[i]}")

    write_train_data_to_table(final_data, write_table_name)


if __name__ == '__main__':
    pass