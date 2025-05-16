import os
from pathlib import Path
from openai import OpenAI
import time
import json
from common import read_data_from_odps

# 初始化客户端
client = OpenAI(
    # 若没有配置环境变量，可用百炼API Key将下行替换为：api_key="sk-xxx"。但不建议在生产环境中直接将API Key硬编码到代码中，以减少API Key泄露风险。
    api_key="xxx",
    base_url="xxx"  # 百炼服务的base_url
)

def messages_builder_example(prompt,url):
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
    return messages

def make_jsonl_file_from_odps(table, model_name='qwen-vl-max', column_id='id', column_prompt='prompt', column_url='content', works=10, start_count=-1, end_count=-1):
    json_file = '_'.join(table.split('/')[-2:]) + '.jsonl'
    df_input = read_data_from_odps(table, "", num_workers=works)
    if start_count >= 0 and end_count >= 0:
        df_input = df_input[start_count:end_count]
    with open(json_file, 'w', encoding='utf-8') as fout:
        for i, row in df_input.iterrows():
            body = {"model": model_name, "messages": messages_builder_example(row[column_prompt], row[column_url]), "temperature": 0.8, "max_tokens": 2048, "headers":{'X-DashScope-DataInspection': '{"input":"disable", "output":"disable"}'}}
            request = {"custom_id": row[column_id], "method": "POST", "url": "/v1/chat/completions", "body": body}
            fout.write(json.dumps(request, separators=(',', ':'), ensure_ascii=False) + "\n", )
    return json_file

def upload_file(file_path):
    print(f"正在上传包含请求信息的JSONL文件...")
    file_object = client.files.create(file=Path(file_path), purpose="batch")
    print(f"文件上传成功。得到文件ID: {file_object.id}\n")
    return file_object.id

def create_batch_job(input_file_id, callback_url=None):
    print(f"正在基于文件ID，创建Batch任务...")
    if callback_url is not None:
        metadata = {
            "ds_batch_finish_callback": callback_url
        }
        batch = client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="336h", metadata=metadata)
    else:
        batch = client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="336h")

    print(f"Batch任务创建完成。 得到Batch任务ID: {batch.id}\n")
    return batch.id

def check_job_status(batch_id):
    print(f"正在检查Batch任务状态...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"Batch任务状态: {batch.status}\n")
    return batch.status

def get_output_id(batch_id):
    print(f"正在获取Batch任务中执行成功请求的输出文件ID...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"输出文件ID: {batch.output_file_id}\n")
    return batch.output_file_id

def get_error_id(batch_id):
    print(f"正在获取Batch任务中执行错误请求的输出文件ID...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"错误文件ID: {batch.error_file_id}\n")
    return batch.error_file_id

def download_results(output_file_id, output_file_path):
    print(f"正在打印并下载Batch任务的请求成功结果...")
    content = client.files.content(output_file_id)
    # 打印部分内容以供测试
    print(f"打印请求成功结果的前1000个字符内容: {content.text[:1000]}...\n")
    # 保存结果文件至本地
    content.write_to_file(output_file_path)
    print(f"完整的输出结果已保存至本地输出文件result.jsonl\n")

def download_errors(error_file_id, error_file_path):
    print(f"正在打印并下载Batch任务的请求失败信息...")
    content = client.files.content(error_file_id)
    # 打印部分内容以供测试
    print(f"打印请求失败信息的前1000个字符内容: {content.text[:1000]}...\n")
    # 保存错误信息文件至本地
    content.write_to_file(error_file_path)
    print(f"完整的请求失败信息已保存至本地错误文件error.jsonl\n")

def start_batch_job(input_file, callback_url=None):
    try:
        # Step 1: 上传包含请求信息的JSONL文件，得到输入文件ID
        input_file_id = upload_file(input_file)
        # Step 2: 基于输入文件ID，创建Batch任务
        batch_id = create_batch_job(input_file_id, callback_url)
        print(f"Batch任务创建成功。 得到Batch任务ID: {batch_id}\n")
        return batch_id
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return None

def check_batch_job_status(batch_id):
    try:
        status = check_job_status(batch_id)
        while status not in ["completed", "failed", "expired", "cancelled"]:
            status = check_job_status(batch_id)
            print(f"等待任务完成...")
            time.sleep(10)  # 等待10秒后再次查询状态
        # 如果任务失败，则打印错误信息并退出
        if status == "failed":
            batch = client.batches.retrieve(batch_id)
            print(f"Batch任务失败。错误信息为:{batch.errors}\n")
            print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            return
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

def download_batch_job_results(batch_id, output_file_path, error_file_path, output_table=None):
    try:
        # Step 4: 下载结果：如果输出文件ID不为空，则打印请求成功结果的前1000个字符内容，并下载完整的请求成功结果到本地输出文件；
        # 如果错误文件ID不为空，则打印请求失败信息的前1000个字符内容，并下载完整的请求失败信息到本地错误文件。
        output_file_id = get_output_id(batch_id)
        if output_file_id:
            download_results(output_file_id, output_file_path)
        error_file_id = get_error_id(batch_id)
        if error_file_id:
            download_errors(error_file_id, error_file_path)
            print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

def main():
    # 文件路径
    input_file_path = "test.jsonl"  # 可替换为您的输入文件路径
    output_file_path = "result.jsonl"  # 可替换为您的输出文件路径
    error_file_path = "error.jsonl"  # 可替换为您的错误文件路径
    try:
        # Step 1: 上传包含请求信息的JSONL文件，得到输入文件ID
        input_file_id = upload_file(input_file_path)
        # Step 2: 基于输入文件ID，创建Batch任务
        batch_id = create_batch_job(input_file_id)
        # Step 3: 检查Batch任务状态直到结束
        status = ""
        while status not in ["completed", "failed", "expired", "cancelled"]:
            status = check_job_status(batch_id)
            print(f"等待任务完成...")
            time.sleep(10)  # 等待10秒后再次查询状态
        # 如果任务失败，则打印错误信息并退出
        if status == "failed":
            batch = client.batches.retrieve(batch_id)
            print(f"Batch任务失败。错误信息为:{batch.errors}\n")
            print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            return
        # Step 4: 下载结果：如果输出文件ID不为空，则打印请求成功结果的前1000个字符内容，并下载完整的请求成功结果到本地输出文件；
        # 如果错误文件ID不为空，则打印请求失败信息的前1000个字符内容，并下载完整的请求失败信息到本地错误文件。
        output_file_id = get_output_id(batch_id)
        if output_file_id:
            download_results(output_file_id, output_file_path)
        error_file_id = get_error_id(batch_id)
        if error_file_id:
            download_errors(error_file_id, error_file_path)
            print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

def odps_image_main(table="", model_name="qwen-vl-max", column_id='id', column_prompt='question', column_url='content'):
    input_file = make_jsonl_file_from_odps(table, model_name=model_name, column_id=column_id, column_prompt=column_prompt, column_url=column_url)
    batch_id = start_batch_job(input_file, None)
    print(f"{table}: {model_name}: batch-id = {batch_id}")
    return batch_id

if __name__ == "__main__":
    batch_id = odps_image_main()