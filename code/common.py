import time
from datetime import datetime
from zoneinfo import ZoneInfo
import json, re
from openai import OpenAI

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

def messages_builder_example_one_shot_image(one_shot_question:str, one_shot_question_url:str, one_shot_answer:str, reasoning_question:str, reasoning_question_url:str):
    dialog_message = []
    dialog_message.append(messages_builder_example(one_shot_question, one_shot_question_url, few_shot=True))
    dialog_message.append({"role": "assistant", "content": one_shot_answer})
    dialog_message.append(messages_builder_example(reasoning_question, reasoning_question_url, few_shot=True))
    return dialog_message

def messages_builder_example_one_shot_text(one_shot_question:str, one_shot_answer:str, reasoning_question:str):
    dialog_message = []
    dialog_message.append({"role": "user", "content": one_shot_question})
    dialog_message.append({"role": "assistant", "content": one_shot_answer})
    dialog_message.append({"role": "user", "content": reasoning_question})
    return dialog_message

def extract_json(s):
    i = s.index('{')
    count = 1
    for j, c in enumerate(s[i + 1:], start=i + 1):
        if c == '}':
            count -= 1
        elif c == '{':
            count += 1
        if count == 0:
            break
    assert (count == 0)  # 检查是否找到最后一个'}'
    return s[i:j + 1]

def call_idealab_and_concat_stream_response_to_whole_str(messages:list, model_name="gpt-4o-0513", enable_thinking=True):
    client = OpenAI(
        api_key="api-key",
        base_url="https://openai/api/openai/v1",
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
    return result,reasoning_content


def remove_repeate_item(whole_datas:list):
    whole_set = set()
    final_datas = []
    for item in whole_datas:
        if item['content'] in whole_set:
            print(f"repeate content={item['content']}")
            continue
        else:
            whole_set.add(item['content'])
            final_datas.append(item)
    return final_datas


def remove_repeate_item_of_allinone(whole_datas:list):
    whole_set = set()
    final_datas = []
    for item in whole_datas:
        no_repeate_str = item['content'] + item['extra']
        if no_repeate_str in whole_set:
            print(f"repeate content={no_repeate_str}")
            continue
        else:
            whole_set.add(no_repeate_str)
            final_datas.append(item)
    return final_datas

def uniform_format_of_options(conclustion:str):
    if not isinstance(conclustion, str):
        conclustion = str(conclustion)
    uppercase_letters = re.findall(r'[A-Z]', conclustion)
    unique_letters = list(dict.fromkeys(uppercase_letters))
    unique_letters.sort()
    result = json.dumps(unique_letters, ensure_ascii=False)
    return result

def call_deepseek_vl2(messages:list):
    from openai import OpenAI
    client = OpenAI(api_key="api-key", base_url="https://api.siliconflow.cn/v1")

    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-vl2",
        messages=messages,
        stream=False,
        temperature=0.8,
        max_tokens=256,
    )
    final_res = response.choices[0].message.content
    return final_res

def handle_one_item_for_deepseekvl2(new_item:dict, current_index:int, total_len:int):
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
        print(f'''NO.{current_index}/{total_len} start to process, current time={bj_time.strftime("%Y-%m-%d %H:%M:%S")}, url={new_item['content']}''')
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
            print(f"NO.{current_index}/{total_len} meet the failed when call the deepseekvl2 api,current response={result}, wait to next round.")
            time.sleep(5)
            continue
    return result




if __name__ == '__main__':
    pass