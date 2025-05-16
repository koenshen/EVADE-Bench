import requests
from common import *
from datasets import Dataset, Image, Features, Value
import json
import os
from all_in_one import detail_prompt, simple_prompt, handle_datas_all_in_one_pt, trans_label_to_letter,zhuangyang_label_mapping,fengxiong_label_mapping,zenggao_label_mapping,suoyin_label_mapping,jianfei_label_mapping,disease_label_mapping

local_image_path = "images/"

def download_images():
    # 读取JSON文件
    with open("need_to_review_risk_from_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    mapping_table_name = "table_name"
    mapping_datas = read_data_from_odps(mapping_table_name)
    oldurl_to_newurl_dict = {}
    for index, item in mapping_datas.iterrows():
        oldurl_to_newurl_dict[item['url']] = item['result']

    human_data_table = "table_name"
    human_datas = get_human_table_datas_to_list(human_data_table)
    set2 = set()
    for index, item in enumerate(human_datas):
        set2.add(item['content'])
    set1 = set(oldurl_to_newurl_dict.keys())

    difference_images = set1^set2

    could_not_found_url = 0
    error_download_url = 0
    # 遍历每个字典
    for index, item in enumerate(data):
        content = item.get('content', '')

        # 检查content是否符合条件
        if content.startswith(('http://', 'https://')):
            if content in oldurl_to_newurl_dict.keys():
                content = oldurl_to_newurl_dict[content]
            else:
                could_not_found_url += 1
                print(f"NO.{could_not_found_url}/{len(data)}, can not find the old url in oss. url={content}")
            try:
                file_ext = content.split(".")[-1]
                file_name = f"{item['id']}_{item['extra']}.{file_ext}"
                save_path = local_image_path + file_name

                response = requests.get(content, timeout=10)
                response.raise_for_status()

                # 保存文件
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"NO.{index+1}/{len(data)}下载成功：{file_name}")

            except Exception as e:
                error_download_url += 1
                print(f"NO.{error_download_url}/{len(data)}, 下载失败（id={item['id']}）: {str(e)}")




def upload_to_huggingface():
    from huggingface_hub import HfApi
    from datasets import Dataset, Features, Value, Image
    import os
    import json

    content_type_mapping = {
        "suoyin":"Women's Health",
        "zhuangyang": "Men's Health",
        "tall": "Height Growth",
        "fengxiong": "Body Shaping",
        "jianfei": "Weight Loss",
        "disease": "Health Supplement",
    }

    single_to_allinone_answer_mapping = {
        "suoyin":suoyin_label_mapping,
        "zhuangyang": zhuangyang_label_mapping,
        "tall": zenggao_label_mapping,
        "fengxiong": fengxiong_label_mapping,
        "jianfei": jianfei_label_mapping,
        "disease": disease_label_mapping,
    }

    # 读取本地 JSON 文件
    json_path = "need_to_review_risk_from_data.json"  # 包含 100 条记录的 JSON 文件路径
    image_dir = "images"  # 存放所有图片的文件夹路径

    # 读取 JSON 文件（假设是一个列表，每项为字典）
    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # 划分数据集为 image 和 text
    image_data = []
    text_data = []
    error_case = 0
    for entry in data_list:
        new_entry = {}
        new_entry["id"] = str(entry.get("id", ""))  # 样本 ID
        content_type = entry.get("content_type", "")
        new_entry["content_type"] = content_type_mapping[content_type]
        options = entry.get("选项", entry.get("options", None))
        if options is not None:
            new_entry["single_risk_options"] = options  # 直接赋值为字符串
        new_entry["extra"] = entry.get("extra", "")

        # 判断 content 字段是否以 http:// 或 https:// 开头
        content_val = entry.get("content", "")
        if content_val.startswith("http://") or content_val.startswith("https://"):
            # 属于 image 数据
            new_entry["single_risk_question"] = entry.get("question", "").split("# 给定信息")[0].strip()  # 问题文本
            new_entry["all_in_one_detail_question"] = detail_prompt.split("# 给定信息")[0].strip()
            new_entry["all_in_one_simple_question"] = simple_prompt.split("# 给定信息")[0].strip()
            new_entry["all_in_one_options"] = trans_label_to_letter(options, single_to_allinone_answer_mapping[content_type])
            file_ext = content_val.split(".")[-1]
            image_rel_path = f"{new_entry['id']}_{new_entry['extra']}.{file_ext}"
            image_path = os.path.join(image_dir, image_rel_path)
            if not os.path.isfile(image_path):
                error_case += 1
                print(f"NO.{error_case} 图片文件不存在: {image_path}, 不上传这张图片")
                continue
            image_data.append({**new_entry, "content_image": image_path, "content_text": None})  # 只填 content_image
        else:
            # 属于 text 数据
            new_entry["single_risk_question"] = entry.get("question", "")
            new_entry["all_in_one_detail_question"] = detail_prompt.replace("<text>", content_val)
            new_entry["all_in_one_simple_question"] = simple_prompt.replace("<text>", content_val)
            new_entry["all_in_one_options"] = trans_label_to_letter(options, single_to_allinone_answer_mapping[content_type])
            text_data.append({**new_entry, "content_image": None, "content_text": content_val})  # 只填 content_text

    # --------- 步骤 2：创建 Hugging Face 数据集对象 ---------
    # 定义数据集字段的类型
    features = Features({
        "id": Value("string"),
        "content_image": Image(),  # 图像列（URL 或路径）
        "content_text": Value("string"),  # 文本列
        "content_type": Value("string"),
        "single_risk_question": Value("string"),
        "all_in_one_detail_question": Value("string"),
        "all_in_one_simple_question": Value("string"),
        "single_risk_options": Value("string"),
        "all_in_one_options": Value("string"),
        "extra": Value("string")
    })

    # 创建两个数据集对象，一个用于图片，一个用于文本
    image_dataset = Dataset.from_list(image_data, features=features)
    text_dataset = Dataset.from_list(text_data, features=features)

    # --------- 步骤 3：推送数据集到 Hugging Face Hub ---------
    try:
        # 上传 image 分片
        # image_dataset.push_to_hub("koenshen/EVADE-Bench", split="image", private=False)
        print("Image 数据集已上传到 Hugging Face Hub, split: image")

        # 上传 text 分片
        # text_dataset.push_to_hub("koenshen/EVADE-Bench", split="text", private=False)
        print("Text 数据集已上传到 Hugging Face Hub, split: text")
    except Exception as e:
        print("推送数据集失败，请检查 Hugging Face Token 或权限:", e)

    # --------- 步骤 4：创建并上传 README.md ---------
    # 使用 huggingface_hub 的 HfApi 上传 README 到数据集仓库
    api = HfApi()
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id="xxxx",
        repo_type="dataset",
        commit_message="Add README"
    )
    print("README.md 已上传至 Hugging Face 数据集仓库")


if __name__ == '__main__':
    # download_images()
    upload_to_huggingface()