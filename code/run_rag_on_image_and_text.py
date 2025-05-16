from common import *
from dashscope_image_generation import odps_image_main
from dashscope_generation import odps_main


image_table_name = "table_name"
text_table_name = "table_name"
new_text_rag_doc_table_name = "table_name"
new_text_rag_test_table_name = "table_name"
new_image_rag_doc_table_name = "table_name"
new_image_rag_test_table_name = "table_name"

image_test_to_doc_mapping_table_name = "table_name"
text_test_to_doc_mapping_table_name = "table_name"

def save_rag_image_and_text_to_clip_search_close_image_and_text():
    rag_doc_rate = 0.2

    image_datas = read_data_from_odps(image_table_name)
    text_datas = read_data_from_odps(text_table_name)

    image_final_datas = []
    text_final_datas = []

    for index, item in image_datas.iterrows():
        new_item  = {}
        for key,value in item.items():
            new_item[key] = value
        image_final_datas.append(new_item)

    random.shuffle(image_final_datas)
    sample_num = int(rag_doc_rate * len(image_final_datas))
    image_doc = image_final_datas[:sample_num]
    image_test = image_final_datas[sample_num:]

    for index, item in text_datas.iterrows():
        new_item  = {}
        for key,value in item.items():
            new_item[key] = value
        text_final_datas.append(new_item)
    random.shuffle(text_final_datas)
    sample_num = int(rag_doc_rate * len(text_final_datas))
    text_doc = text_final_datas[:sample_num]
    text_test = text_final_datas[sample_num:]

    write_train_data_to_table(image_doc, new_image_rag_doc_table_name)
    write_train_data_to_table(image_test, new_image_rag_test_table_name)
    write_train_data_to_table(text_doc, new_text_rag_doc_table_name)
    write_train_data_to_table(text_test, new_text_rag_test_table_name)

def handle_image_rag():
    doc_datas = read_data_from_odps(new_image_rag_doc_table_name)
    test_datas = read_data_from_odps(new_image_rag_test_table_name)
    mapping_datas = read_data_from_odps(image_test_to_doc_mapping_table_name)

    doc_id_dict = {}
    for index, item in doc_datas.iterrows():
        new_item = {}
        for key, value in item.items():
            new_item[key] = value
        test_id = str(new_item['id'])
        doc_id_dict[test_id] = new_item

    test_id_dict = {}
    for index, item in test_datas.iterrows():
        new_item = {}
        for key, value in item.items():
            new_item[key] = value
        test_id = str(new_item['id'])
        test_id_dict[test_id] = new_item

    mapping_dict = {}
    for index, item in mapping_datas.iterrows():
        new_item = {}
        for key, value in item.items():
            new_item[key] = value
        test_id = new_item['pk']
        doc_id = new_item['knn_result']
        doc_item = doc_id_dict[doc_id]
        mapping_dict[test_id] = doc_item

    final_data = []
    for test_id, doc_item in mapping_dict.items():
        test_item = test_id_dict[test_id]
        one_shot_prompt_json = messages_builder_example_one_shot_image(one_shot_question=doc_item['question'], one_shot_question_url=doc_item['content'], one_shot_answer=doc_item['选项'], reasoning_question=test_item['question'], reasoning_question_url=test_item['content'])
        test_item['question'] = json.dumps(one_shot_prompt_json, ensure_ascii=False, indent=2)
        final_data.append(test_item)

    with open("rag_datas/allinone_image.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_data, ensure_ascii=False, indent=2))

def handle_text_rag():
    doc_datas = read_data_from_odps(new_text_rag_doc_table_name)
    test_datas = read_data_from_odps(new_text_rag_test_table_name)
    mapping_datas = read_data_from_odps(text_test_to_doc_mapping_table_name)

    doc_id_dict = {}
    for index, item in doc_datas.iterrows():
        new_item = {}
        for key,value in item.items():
            new_item[key] = value
        test_id = str(new_item['id'])
        doc_id_dict[test_id] = new_item

    test_id_dict = {}
    for index, item in test_datas.iterrows():
        new_item = {}
        for key,value in item.items():
            new_item[key] = value
        test_id = str(new_item['id'])
        test_id_dict[test_id] = new_item

    mapping_dict = {}
    for index, item in mapping_datas.iterrows():
        new_item = {}
        for key,value in item.items():
            new_item[key] = value
        test_id = new_item['pk']
        doc_id = new_item['knn_result']
        doc_item = doc_id_dict[doc_id]
        mapping_dict[test_id] = doc_item

    final_data = []
    for test_id, doc_item in mapping_dict.items():
        test_item = test_id_dict[test_id]
        one_shot_prompt_json = messages_builder_example_one_shot_text(one_shot_question=doc_item['question'], one_shot_answer=doc_item['选项'], reasoning_question=test_item['question'])
        test_item['question'] = json.dumps(one_shot_prompt_json, ensure_ascii=False, indent=2)
        # call_idealab_and_concat_stream_response_to_whole_str(one_shot_prompt_json)
        final_data.append(test_item)

    with open("rag_datas/allinone_text.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_data, ensure_ascii=False, indent=2))

def run_text_rag():
    with open("rag_datas/allinone_text.json", 'r', encoding='utf-8') as f:
        datas = json.load(f)

    omega_need_to_fix_table_name =   "table_name"
    write_train_data_to_omega_table(datas, omega_need_to_fix_table_name, few_shot=True)

    bailian_need_to_fix_table_name = "table_name"
    write_train_data_to_table(datas, bailian_need_to_fix_table_name)

    # allinone_text_rag
    # batch_e83880fd-2f86-48d6-87d5-516d1d32ae1f
    odps_main(table=bailian_need_to_fix_table_name)

def run_image_rag():
    with open("rag_datas/allinone_image.json", 'r', encoding='utf-8') as f:
        datas = json.load(f)


    omega_need_to_fix_table_name =   "table_name"
    write_train_data_to_omega_table(datas, omega_need_to_fix_table_name, few_shot=True)

    bailian_need_to_fix_table_name = "table_name"
    write_train_data_to_table(datas, bailian_need_to_fix_table_name)

    # allinone_image_rag
    # batch_95fdf41a-d3e6-42b1-80df-73f6cbae03e7
    odps_main(table=bailian_need_to_fix_table_name)

def read_text_rag_and_compare_accuracy():
    text_rag_table_name = "odps://alimama_intern_dev/tables/yzh_risk_large_model_data_label_eval_info/pt=allinone_text_rag"
    original_table_name = get_human_table_datas_to_list(f"{text_rag_table_name}/model=human")

    def concat_deepseekv3_text(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_deepseekv3_result"
        write_table_name = f"{text_rag_table_name}/model=deepseek-v3"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_deepseekr1_text(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_deepseekr1_result"
        write_table_name = f"{text_rag_table_name}/model=deepseek-r1"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_llama8b_text(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_llama8b_result"
        write_table_name = f"{text_rag_table_name}/model=llama3.1-8b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_llama70b_text(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_llama70b_result"
        write_table_name = f"{text_rag_table_name}/model=llama3.1-70b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_qwen7b_text(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen25_7b_result"
        write_table_name = f"{text_rag_table_name}/model=qwen2.5-7b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_qwen14b_text(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen25_14b_result"
        write_table_name = f"{text_rag_table_name}/model=qwen2.5-14b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_qwen32b_text(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen25_32b_result"
        write_table_name = f"{text_rag_table_name}/model=qwen2.5-32b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_qwen72b_text(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen25_72b_result"
        write_table_name = f"{text_rag_table_name}/model=qwen2.5-72b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_qwenmax_text(original_table_name):
        data_file_path = "bailian_data/allinone_text_rag-qwen-max-batch_e83880fd-2f86-48d6-87d5-516d1d32ae1f.jsonl"
        write_table_name = f"{text_rag_table_name}/model=qwenmax"
        read_bailian_jsonl_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, data_file_path=data_file_path)

    def concat_omega_qwen3_32b_text(original_table_name):
        omega_output_table_name =  "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen3_32b_think_false_result"
        write_table_name = f"{text_rag_table_name}/model=Qwen3-32B_enable_thinking_False"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_omega_qwen3_30b_text(original_table_name):
        omega_output_table_name =  "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen3_30b_moe_think_false_result"
        write_table_name = f"{text_rag_table_name}/model=Qwen3-30B-A3B_enable_thinking_False"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_omega_qwen3_235b_text(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen3_235b_moe_think_false_result"
        write_table_name = f"{text_rag_table_name}/model=Qwen3-235B-A22B_enable_thinking_False"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_omega_qwen3_32b_think_text(original_table_name):
        omega_output_table_name =  "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen3_32b_think_true_result"
        write_table_name = f"{text_rag_table_name}/model=Qwen3-32B_enable_thinking_True"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_omega_qwen3_30b_think_text(original_table_name):
        omega_output_table_name =  "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen3_30b_moe_think_true_result"
        write_table_name = f"{text_rag_table_name}/model=Qwen3-30B-A3B_enable_thinking_True"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_omega_qwen3_235b_think_text(original_table_name):
        omega_output_table_name =  "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_text_rag_qwen3_235b_moe_think_true_result"
        write_table_name = f"{text_rag_table_name}/model=Qwen3-235B-A22B_enable_thinking_True"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    concat_qwen7b_text(original_table_name)
    concat_qwen14b_text(original_table_name)
    concat_qwen32b_text(original_table_name)
    concat_qwen72b_text(original_table_name)
    concat_llama8b_text(original_table_name)
    concat_llama70b_text(original_table_name)
    concat_deepseekv3_text(original_table_name)
    concat_deepseekr1_text(original_table_name)
    concat_qwenmax_text(original_table_name)
    concat_omega_qwen3_32b_text(original_table_name)
    concat_omega_qwen3_30b_text(original_table_name)
    concat_omega_qwen3_235b_text(original_table_name)
    concat_omega_qwen3_32b_think_text(original_table_name)
    concat_omega_qwen3_30b_think_text(original_table_name)
    concat_omega_qwen3_235b_think_text(original_table_name)

def read_image_rag_and_compare_accuracy():
    
    image_rag_table_name = "odps://alimama_intern_dev/tables/yzh_risk_large_model_data_label_eval_info/pt=allinone_image_rag_v2"
    original_table_name = get_human_table_datas_to_list(f"{image_rag_table_name}/model=human")

    def concat_minicpmv_image(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_image_rag_minicpmv8b_result"
        write_table_name = f"{image_rag_table_name}/model=minicpmv"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_internvl8b_image(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_image_rag_internvl8b_result"
        write_table_name = f"{image_rag_table_name}/model=internvl8b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_internvl14b_image(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_image_rag_internvl14b_result"
        write_table_name = f"{image_rag_table_name}/model=internvl14b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_internvl38b_image(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_image_rag_internvl38b_result"
        write_table_name = f"{image_rag_table_name}/model=internvl38b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_qwenvl72b_image(original_table_name):
        omega_output_table_name = "odps://alimama_intern_dev/tables/acx_alimm_benchmark_run_on_small_model/ds=allinone_image_rag_qwen25vl72b_result"
        write_table_name = f"{image_rag_table_name}/model=qwenvl72b"
        read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

    def concat_qwenvlmax_image(original_table_name):
        data_file_path = "bailian_data/allinone_image_rag-qwen-vl-max-batch_95fdf41a-d3e6-42b1-80df-73f6cbae03e7.jsonl"
        write_table_name = f"{image_rag_table_name}/model=qwenvlmax"
        read_bailian_jsonl_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, data_file_path=data_file_path)

    # concat_minicpmv_image(original_table_name)
    concat_internvl8b_image(original_table_name)
    concat_internvl14b_image(original_table_name)
    concat_internvl38b_image(original_table_name)
    concat_qwenvl72b_image(original_table_name)
    # concat_qwenvlmax_image(original_table_name)

if __name__ == '__main__':
    # handle_text_rag()
    # handle_image_rag()
    # run_text_rag()
    # run_image_rag()
    # read_text_rag_and_compare_accuracy()
    read_image_rag_and_compare_accuracy()