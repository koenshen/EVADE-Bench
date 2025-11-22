from utils import *
from common_instruct import *

repeate_time = 50
NUM_THREADS = 50

# model_name_vlm = "gpt-4o-0806"
# model_name_vlm = "qwen2.5-vl-7b-instruct" # bailian
# model_name_vlm = "claude37_sonnet"
model_name_vlm = "gemini-2.5-pro"

model_name_llm = "qwen2.5-72b-instruct"
vlm_instruct_name = "vlm_prompt_251120_v5"
llm_instruct_name = "llm_prompt_251120_v6"

def call_api(prompt:str, image_url:str, model_name:str, is_thinking=False):
    return call_idealab_api(prompt, image_url, model_name)

def process_rows(thread_id, rows, save_path_template):
    result_json = []
    row_num = 0

    # 为每个线程创建独立的保存路径
    thread_save_path = save_path_template.replace('.json', f'_thread_{thread_id}.json')

    for _, row in rows.iterrows():
        row_num += 1
        result_dict = {}
        result_dict['id'] = row['id']
        result_dict['ground_truth'] = row['single_risk_options']
        result_dict['content_type'] = row['content_type']
        # generation 推理
        for _ in range(repeate_time):
            try:
                original_prompt = row['single_risk_question']
                data_uri = concat_base64_image_url(row['content_image'])

                pre_handle_image_to_text_prompt = super_vlm_instruct_of_decompose_version_dict[vlm_instruct_name][row['content_type']] # 当vlm prompt是需要根据content_type定制时
                # pre_handle_image_to_text_prompt = super_vlm_instruct_of_decompose_version_dict[vlm_instruct_name] # 当vlm prompt在6类中通用
                result_vlm, reasoning_content = call_idealab_api_without_stream(prompt=pre_handle_image_to_text_prompt, image_url=data_uri, model_name=model_name_vlm)
                if is_limit_api(result_vlm):
                    continue

                complex_rules = extract_complex_rules_from_prompt(original_prompt)
                prompt = super_llm_instruct_of_decompose_version_dict[llm_instruct_name].replace("${result_vlm}", result_vlm).replace("${rules}", complex_rules)
                result_llm, reasoning_content = call_idealab_api(prompt=prompt, image_url="", model_name=model_name_llm)

                if is_limit_api(result_llm):
                    continue
                if not validate_and_extract_box_content(result_llm):
                    continue

                result_dict['vlm_prompt'] = pre_handle_image_to_text_prompt
                result_dict['llm_prompt'] = prompt
                result_dict['vlm_prompt_name'] = vlm_instruct_name
                result_dict['llm_prompt_name'] = llm_instruct_name
                result_dict['model_name_vlm'] = model_name_vlm
                result_dict['model_name_llm'] = model_name_llm
                result_dict['result_vlm'] = result_vlm
                result_dict['generate_results'] = result_llm
                print(f"Thread {thread_id}: NO.{row_num}/{len(rows)} of generation")
                print("-" * 100)
                break
            except Exception as e:
                print(f"Thread {thread_id}: Error - {str(e)}")
                print(f"There is error on NO.{_ + 1} time, continue to next row.")
                time.sleep(1)

        print("X" * 300)
        result_json.append(result_dict)
        # 每处理完一行就保存一次
        with open(thread_save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_json, ensure_ascii=False, indent=2))


def merge_results(save_path_template, final_save_path):
    merged_results = []

    # 读取并合并所有线程的结果
    for i in range(NUM_THREADS):
        thread_save_path = save_path_template.replace('.json', f'_thread_{i}.json')
        if os.path.exists(thread_save_path):
            with open(thread_save_path, 'r', encoding='utf-8') as f:
                thread_results = json.load(f)
                merged_results.extend(thread_results)
            # 可选：删除临时文件
            os.remove(thread_save_path)

    # 保存合并后的结果
    with open(final_save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(merged_results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 加载数据集
    dataset_name = 'koenshen/EVADE-Bench'
    image_test_dataset = datasets.load_dataset(dataset_name, split="image_test")
    # 转换为DataFrame
    image_test_datas = pd.DataFrame(image_test_dataset)

    # 创建保存路径
    timestamp = time.strftime('%y%m%d%H%M%S')
    save_path_template = f"../data/image_test-decompose-{timestamp}-{model_name_llm}.json"
    final_save_path = save_path_template

    # 将数据平均分配给各个线程
    total_rows = len(image_test_datas)
    rows_per_thread = total_rows // NUM_THREADS
    remainder = total_rows % NUM_THREADS  # 计算余数
    threads = []

    for i in range(NUM_THREADS):
        start_idx = i * rows_per_thread + min(i, remainder)
        end_idx = start_idx + rows_per_thread + (1 if i < remainder else 0)
        thread_df = image_test_datas.iloc[start_idx:end_idx]
        print(f"Thread {i}: 分配 {len(thread_df)} 条数据")

        thread = threading.Thread(
            target=process_rows,
            args=(i, thread_df, save_path_template)
        )
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 合并所有线程的结果
    merge_results(save_path_template, final_save_path)
    print("All processing completed and results merged.")

    with open(save_path_template, 'r', encoding="utf-8") as f:
        result_list = json.load(f)
    get_inference_result_and_check_accuracy(result_list)
