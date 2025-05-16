import random

from common import *
from dashscope_generation import odps_main

whole_pt_name = "odps://alimama_intern_dev/tables/yzh_risk_large_model_data_label_eval_info/pt=whole_text_single_risk_250509_v7"

def concat_human_text(whole_text_table_name:str):
    fengxiong_table = "table_name"
    zhuangyang_table = "table_name"
    disease_table ="table_name"
    tall_table = "table_name"
    suoyin_table = "table_name"
    jianfei_table = "table_name"

    fengxiong_datas = read_data_from_odps(fengxiong_table)
    zhuangyang_datas = read_data_from_odps(zhuangyang_table)
    disease_datas = read_data_from_odps(disease_table)
    tall_datas = read_data_from_odps(tall_table)
    suoyin_datas = read_data_from_odps(suoyin_table)
    jianfei_datas = read_data_from_odps(jianfei_table)

    def concat_six_risk_text(whole_text_table_name: str, fengxiong_datas, zhuangyang_datas, disease_datas, tall_datas, suoyin_datas, jianfei_datas):
        fengxiong_final_datas = []
        zhuangyang_final_datas = []
        disease_final_datas = []
        tall_final_datas = []
        suoyin_final_datas = []
        jianfei_final_datas = []

        fengxiong_start_id = 1000000
        for index, item in fengxiong_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value

            new_item['extra'] = str(copy.deepcopy(new_item['id']))
            fengxiong_start_id += 1
            new_item['id'] = fengxiong_start_id

            new_item = label_contenttype_by_prompt(new_item)
            fengxiong_final_datas.append(new_item)

        zhuangyang_start_id = 2000000
        for index, item in zhuangyang_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value

            new_item['extra'] = str(copy.deepcopy(new_item['id']))
            zhuangyang_start_id += 1
            new_item['id'] = zhuangyang_start_id

            new_item = label_contenttype_by_prompt(new_item)
            zhuangyang_final_datas.append(new_item)

        disease_start_id = 3000000
        for index, item in disease_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value

            new_item['extra'] = str(copy.deepcopy(new_item['id']))
            disease_start_id += 1
            new_item['id'] = disease_start_id

            new_item = label_contenttype_by_prompt(new_item)
            disease_final_datas.append(new_item)

        tall_start_id = 4000000
        for index, item in tall_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value

            new_item['extra'] = str(copy.deepcopy(new_item['id']))
            tall_start_id += 1
            new_item['id'] = tall_start_id

            new_item = label_contenttype_by_prompt(new_item)
            tall_final_datas.append(new_item)

        suoyin_start_id = 5000000
        for index, item in suoyin_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value

            new_item['extra'] = str(copy.deepcopy(new_item['id']))
            suoyin_start_id += 1
            new_item['id'] = suoyin_start_id

            new_item = label_contenttype_by_prompt(new_item)
            suoyin_final_datas.append(new_item)

        whole_text_datas = fengxiong_final_datas + zhuangyang_final_datas + disease_final_datas + tall_final_datas + suoyin_final_datas
        new_whole_text_datas = remove_repeate_item(whole_text_datas)

        omega_input_datas = read_data_from_odps("table_name")
        omega_input_list = []
        for index, omega_item in omega_input_datas.iterrows():
            new_item = {}
            for key,value in omega_item.items():
                new_item[key] = value
            omega_input_list.append(new_item)
        new_whole_text_datas_remove_inconsistent = check_concathumandata_is_consistent_with_omegainput(new_whole_text_datas, omega_input_list)

        # 因为减肥的文本不在omega中,所以如果直接对包含减肥的所有文本直接check_concathumandata_is_consistent_with_omegainput就会丢失减肥的文本,因此需要在最后补上
        jianfei_start_id = 6000000
        for index, item in jianfei_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value

            new_item['extra'] = str(copy.deepcopy(new_item['id']))
            jianfei_start_id += 1
            new_item['id'] = jianfei_start_id

            new_item = label_contenttype_by_prompt(new_item)
            jianfei_final_datas.append(new_item)

        # 补上减肥的文本后,以防万一,再去一次重
        jianfei_final_datas_remove_repeate = remove_repeate_item(jianfei_final_datas)
        new_whole_text_datas_remove_inconsistent += jianfei_final_datas_remove_repeate
        new_whole_text_datas_remove_inconsistent = remove_repeate_item(new_whole_text_datas_remove_inconsistent)


        random.shuffle(new_whole_text_datas_remove_inconsistent)
        write_train_data_to_table(new_whole_text_datas_remove_inconsistent, whole_text_table_name)

    concat_six_risk_text(whole_text_table_name=whole_text_table_name,
                          fengxiong_datas=fengxiong_datas, 
                          zhuangyang_datas=zhuangyang_datas, 
                          disease_datas=disease_datas, 
                          tall_datas=tall_datas, 
                          suoyin_datas=suoyin_datas,
                          jianfei_datas=jianfei_datas)

def concat_deepseekv3_text(original_table_name):
    omega_output_table_name = "table_name"
    write_table_name = f"{whole_pt_name}/model=deepseek-v3"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_deepseekr1_text(original_table_name):
    omega_output_table_name = "table_name"
    write_table_name = f"{whole_pt_name}/model=deepseek-r1"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_llama8b_text(original_table_name):
    omega_output_table_name = "table_name"
    write_table_name = f"{whole_pt_name}/model=llama3.1-8b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_llama70b_text(original_table_name):
    omega_output_table_name = "table_name"
    write_table_name = f"{whole_pt_name}/model=llama3.1-70b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_llama405b_text(original_table_name):
    omega_output_table_name = "table_name"
    write_table_name = f"{whole_pt_name}/model=llama3.1-405b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_qwen7b_text(original_table_name):
    omega_output_table_name =  "table_name"
    write_table_name = f"{whole_pt_name}/model=qwen2.5-7b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_qwen14b_text(original_table_name):
    omega_output_table_name =  "table_name"
    write_table_name = f"{whole_pt_name}/model=qwen2.5-14b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_qwen32b_text(original_table_name):
    omega_output_table_name =  "table_name"
    write_table_name = f"{whole_pt_name}/model=qwen2.5-32b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_qwen72b_text(original_table_name):
    omega_output_table_name =  "table_name"
    write_table_name = f"{whole_pt_name}/model=qwen2.5-72b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_qwenmax_text(original_table_name):
    data_file_path = "table_name"
    write_table_name = f"{whole_pt_name}/model=qwenmax"
    read_bailian_jsonl_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, data_file_path=data_file_path)


def concat_o1mini_text(original_table_name):
    write_table_name = f"{whole_pt_name}/model=o1mini"
    model_name = "o1-mini-0912"
    fengxiong_table =  "table_name"
    zhuangyang_table = "table_name"
    disease_table =    "table_name"
    tall_table =       "table_name"
    suoyin_table =     "table_name"
    jianfei_table =    "table_name"
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=fengxiong_table, model_name=model_name, use_old_extra_id=True, risk_type="fengxiong")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=zhuangyang_table, model_name=model_name, use_old_extra_id=True, risk_type="zhuangyang")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=disease_table, model_name=model_name, use_old_extra_id=True, risk_type="disease")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=tall_table, model_name=model_name, use_old_extra_id=True, risk_type="tall")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=suoyin_table, model_name=model_name, use_old_extra_id=True, risk_type="2-suoyin")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=jianfei_table, model_name=model_name, use_old_extra_id=True, risk_type="jianfei")


def concat_gpt41_text(original_table_name):
    write_table_name = f"{whole_pt_name}/model=gpt41"
    model_name = "gpt-41-0414-global"
    fengxiong_table =  "table_name"
    zhuangyang_table = "table_name"
    disease_table =    "table_name"
    tall_table =       "table_name"
    suoyin_table =     "table_name"
    jianfei_table =    "table_name"
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=fengxiong_table, model_name=model_name, use_old_extra_id=True, risk_type="fengxiong")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=zhuangyang_table, model_name=model_name, use_old_extra_id=True, risk_type="zhuangyang")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=disease_table, model_name=model_name, use_old_extra_id=True, risk_type="disease")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=tall_table, model_name=model_name, use_old_extra_id=True, risk_type="tall")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=suoyin_table, model_name=model_name, use_old_extra_id=True, risk_type="2-suoyin")
    read_idealab_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, idealab_table_name=jianfei_table, model_name=model_name, use_old_extra_id=True, risk_type="jianfei")


def concat_qwen3_text(original_table_name):
    model_name_list = ["Qwen3-30B-A3B", "Qwen3-235B-A22B", "Qwen3-32B"]
    enable_thinking_list = [False, True]

    # 可以自己选择传入的是原始table名或者传入一个list,如果传入list可以节省很大一部分的读表时间
    original_list = []
    if isinstance(original_table_name,str):
        original_datas = read_data_from_odps(original_table_name)
        # 组装原始数据和回答
        for index, item in original_datas.iterrows():
            new_item = {}
            for key, value in item.items():
                new_item[key] = value
            original_list.append(new_item)

    if isinstance(original_table_name, list):
        original_list = copy.deepcopy(original_table_name)

    for model_name in model_name_list:
        for enable_thinking in enable_thinking_list:
            final_data = []
            error_case = 0
            new_table_name = f"{whole_pt_name}/model={model_name}_enable_thinking_{enable_thinking}"
            with open(f"qwen3_datas_whole_risk_text/{model_name}/enable_thinking_{enable_thinking}/whole_result.json", 'r', encoding='utf-8') as f:
                datas = json.load(f)
            with open(f"qwen3_datas_jianfei/{model_name}/enable_thinking_{enable_thinking}/whole_result.json", 'r', encoding='utf-8') as f:
                datas_jf = json.load(f)

            id_answer_dict = {}
            for index, item in enumerate(datas):
                id_answer_dict[str(item['id'])] = item['model_response']
            for index, item in enumerate(datas_jf):
                id_answer_dict[str(item['id'])] = item['model_response']

            for index, item in enumerate(original_list):
                id = str(item['id'])
                if item['content_type'] == 'jianfei':
                    id = item['extra']
                if id not in id_answer_dict.keys():
                    print(f"can not find the {id} in the qwen3 results.")
                    continue
                final_data, error_case = handle_single_item_and_extract_analysis_and_conclustion(final_data=final_data, id_answer_dict=id_answer_dict,new_item=item, id_key=id, error_case=error_case)
            print(f"final_datas length={len(final_data)}")
            write_train_data_to_table(final_data, new_table_name)


def handle_text_from_six_risks():
    original_table_name = f"{whole_pt_name}/model=human"
    # concat_human_text(original_table_name)

    human_data = get_human_table_datas_to_list(original_table_name)
    # concat_qwen7b_text(human_data)
    # concat_qwen14b_text(human_data)
    # concat_qwen32b_text(human_data)
    # concat_qwen72b_text(human_data)
    # concat_llama8b_text(human_data)
    # concat_llama70b_text(human_data)
    # concat_deepseekv3_text(human_data)
    # concat_deepseekr1_text(human_data)

    # concat_o1mini_text(human_data)
    # concat_gpt41_text(human_data)

    # concat_qwenmax_text(human_data)
    # concat_qwen3_text(human_data)

    # concat_llama405b_text(original_table_name)


if __name__ == '__main__':
    handle_text_from_six_risks()