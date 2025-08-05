from utils import *

repeate_time = 100
NUM_THREADS = 100

model_name_use = "qwen2.5-7b-instruct"
think_mode = False

def call_api(prompt:str, model_name:str, think_mode=False):
    if "qwen3" in model_name:
        return call_qwen3_api(prompt, model_name, think_mode)
    return call_idealab_api(prompt=prompt, image_url="", model_name=model_name)

def process_rows(thread_id, rows, save_path_template):
    result_json = []
    row_num = 0

    # 为每个线程创建独立的保存路径
    thread_save_path = save_path_template.replace('.json', f'_thread_{thread_id}.json')

    for _, row in enumerate(rows):
        row_num += 1
        result_dict = {}
        result_dict['id'] = row['id']
        result_dict['ground_truth'] = row['all_in_one_options']

        # generation 推理
        for _ in range(repeate_time):
            try:
                prompt = row['all_in_one_detail_question']
                result, reasoning_content = call_api(prompt=prompt, model_name=model_name_use, think_mode=think_mode)

                if not validate_and_extract_three_json(result):
                    continue
                result_dict['prompt'] = prompt
                result_dict['model_name'] = model_name_use
                result_dict['think_mode'] = think_mode
                result_dict['generate_results'] = result
                print(f"Thread {thread_id}: NO.{row_num} of generation")
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
    text_dataset = datasets.load_dataset(dataset_name, split="text")
    # 转换为DataFrame
    text_datas = pd.DataFrame(text_dataset)

    prompt_list = text_datas['content_text'].tolist()
    selected_columns = ['id', 'content_text', 'all_in_one_detail_question', 'all_in_one_options']  # 你想要的列名
    reference_list = text_datas[selected_columns].apply(lambda x: x.to_dict(), axis=1).tolist()
    reference_dict = {}
    for item in reference_list:
        reference_dict[item['content_text']] = item

    doc_datas, test_datas = split_rag_dataset(prompt_list, document_ratio=0.2)

    final_data = []
    print("start process all-in-one rag prompt......")
    for index, query_str in enumerate(test_datas):
        similar_text = retrieve_similar_document(doc_datas, query_str)
        whole_item = reference_dict[similar_text]
        one_shot_prompt_json = messages_builder_example_one_shot_text(one_shot_question=whole_item['all_in_one_detail_question'], one_shot_answer=whole_item['all_in_one_options'], reasoning_question=query_str)
        item = {}
        item['id'] = reference_dict[query_str]['id']
        item['all_in_one_options'] = reference_dict[query_str]['all_in_one_options']
        item['all_in_one_detail_question'] = one_shot_prompt_json
        final_data.append(item)
    print("end process all-in-one rag prompt.")

    # 创建保存路径
    timestamp = time.strftime('%y%m%d%H%M%S')
    save_path_template = f"../datas/rag_test_allinone-{timestamp}.json"
    final_save_path = save_path_template

    # 将数据平均分配给各个线程
    rows_per_thread = len(final_data) // NUM_THREADS
    threads = []

    # 创建并启动线程
    for i in range(NUM_THREADS):
        start_idx = i * rows_per_thread
        end_idx = start_idx + rows_per_thread if i < NUM_THREADS - 1 else len(final_data)
        thread_df = final_data[start_idx:end_idx]

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

    with open(final_save_path, 'r', encoding="utf-8") as f:
        result_list = json.load(f)
    get_inference_result_and_check_accuracy(result_list)
