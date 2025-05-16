from common import *
from dashscope_image_generation import odps_image_main

whole_pt_name = "odps://alimama_intern_dev/tables/yzh_risk_large_model_data_label_eval_info/pt=whole_image_single_risk_250509_v1"

def concat_human_image(whole_image_table_name:str):
    fengxiong_table =  "table_name"
    zhuangyang_table = "table_name"
    disease_table =    "table_name"
    tall_table =       "table_name"
    suoyin_table =    "table_name"
    jianfei_table =    "table_name"

    fengxiong_datas = read_data_from_odps(fengxiong_table)
    zhuangyang_datas = read_data_from_odps(zhuangyang_table)
    disease_datas = read_data_from_odps(disease_table)
    tall_datas = read_data_from_odps(tall_table)
    suoyin_datas = read_data_from_odps(suoyin_table)
    jianfei_datas = read_data_from_odps(jianfei_table)
    
    def concat_six_risk_image(whole_image_table_name: str, fengxiong_datas, zhuangyang_datas, disease_datas, tall_datas, suoyin_datas, jianfei_datas):
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

        whole_image_datas = fengxiong_final_datas + zhuangyang_final_datas + disease_final_datas + tall_final_datas + suoyin_final_datas + jianfei_final_datas
        new_whole_image_datas = remove_repeate_item(whole_image_datas)
        omega_input_datas = read_data_from_odps("table_name")
        omega_input_list = []
        for index, omega_item in omega_input_datas.iterrows():
            new_item = {}
            for key,value in omega_item.items():
                new_item[key] = value
            omega_input_list.append(new_item)
        new_whole_image_datas_remove_inconsistent = check_concathumandata_is_consistent_with_omegainput(new_whole_image_datas, omega_input_list)

        random.shuffle(new_whole_image_datas_remove_inconsistent)
        write_train_data_to_table(new_whole_image_datas_remove_inconsistent, whole_image_table_name)

    concat_six_risk_image(whole_image_table_name=whole_image_table_name,
                          fengxiong_datas=fengxiong_datas,
                          zhuangyang_datas=zhuangyang_datas,
                          disease_datas=disease_datas,
                          tall_datas=tall_datas,
                          suoyin_datas=suoyin_datas,
                          jianfei_datas=jianfei_datas)

def concat_llava16_image(original_table_name:str):
    omega_output_table_name ="table_name"
    write_table_name = f"{whole_pt_name}/model=llava16"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_minicpmv_image(original_table_name:str):
    omega_output_table_name ="table_name"
    write_table_name = f"{whole_pt_name}/model=minicpmv"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_internvl8b_image(original_table_name:str):
    omega_output_table_name = "table_name"
    write_table_name = f"{whole_pt_name}/model=internvl8b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_internvl14b_image(original_table_name:str):
    omega_output_table_name = "table_name"
    write_table_name = f"{whole_pt_name}/model=internvl14b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_internvl38b_image(original_table_name:str):
    omega_output_table_name ="table_name"
    write_table_name = f"{whole_pt_name}/model=internvl38b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_qwenvl7b_image(original_table_name:str):
    omega_output_table_name ="table_name"
    write_table_name = f"{whole_pt_name}/model=qwenvl7b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_qwenvl32b_image(original_table_name:str):
    omega_output_table_name = "table_name"
    write_table_name = f"{whole_pt_name}/model=qwenvl32b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_qwenvl72b_image(original_table_name:str):
    omega_output_table_name = "table_name"
    write_table_name = f"{whole_pt_name}/model=qwenvl72b"
    read_omega_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, omega_output_table_name=omega_output_table_name)

def concat_qwenvlmax_image(original_table_name:str):
    data_file_path = "table_name"
    write_table_name = f"{whole_pt_name}/model=qwenvlmax"
    read_bailian_jsonl_and_concat_final_data(original_table_name=original_table_name, write_table_name=write_table_name, data_file_path=data_file_path)

def concat_gpt4o_image(original_table_datas:list):
    write_table_name = f"{whole_pt_name}/model=gpt4o"
    model_name = "gpt-4o-0806"
    fengxiong_table =  "table_name"
    zhuangyang_table = "table_name"
    disease_table =    "table_name"
    tall_table =       "table_name"
    suoyin_table =    "table_name"
    jianfei_table =    "table_name"
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=fengxiong_table, model_name=model_name, use_old_extra_id=True, risk_type="fengxiong")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=zhuangyang_table, model_name=model_name, use_old_extra_id=True, risk_type="zhuangyang")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=disease_table, model_name=model_name, use_old_extra_id=False, risk_type="disease")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=tall_table, model_name=model_name, use_old_extra_id=True, risk_type="tall")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=suoyin_table, model_name=model_name, use_old_extra_id=True, risk_type="2-suoyin-202and0.75.sql")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=jianfei_table, model_name=model_name, use_old_extra_id=True, risk_type="jianfei")

def concat_claude37_image(original_table_datas:list):
    write_table_name = f"{whole_pt_name}/model=claude37"
    model_name = "claude37_sonnet"
    fengxiong_table =  "table_name"
    zhuangyang_table = "table_name"
    disease_table =    "table_name"
    tall_table =       "table_name"
    suoyin_table =    "table_name"
    jianfei_table =    "table_name"
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=fengxiong_table, model_name=model_name, use_old_extra_id=True, risk_type="fengxiong")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=zhuangyang_table, model_name=model_name, use_old_extra_id=True, risk_type="zhuangyang")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=disease_table, model_name=model_name, use_old_extra_id=False, risk_type="disease")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=tall_table, model_name=model_name, use_old_extra_id=True, risk_type="tall")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=suoyin_table, model_name=model_name, use_old_extra_id=True, risk_type="2-suoyin-202and0.75.sql")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=jianfei_table, model_name=model_name, use_old_extra_id=True, risk_type="jianfei")

def concat_gemini25pro_image(original_table_datas:list):
    write_table_name = f"{whole_pt_name}/model=gemini25pro"
    model_name = "gemini-2.5-pro"
    fengxiong_table =  "table_name"
    zhuangyang_table = "table_name"
    disease_table =    "table_name"
    tall_table =       "table_name"
    suoyin_table =    "table_name"
    jianfei_table =    "table_name"

    # 只有gemini25pro的疾病需要使用旧id,因为它是以前先汇总后再跑的,所以idealab的output存储的是旧id,需要用extra,其余的两个都重跑了所以不用.
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=disease_table, model_name=model_name, use_old_extra_id=True, risk_type="disease")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=fengxiong_table, model_name=model_name, use_old_extra_id=True, risk_type="fengxiong")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=zhuangyang_table, model_name=model_name, use_old_extra_id=True, risk_type="zhuangyang")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=tall_table, model_name=model_name, use_old_extra_id=True, risk_type="tall")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=suoyin_table, model_name=model_name, use_old_extra_id=True, risk_type="2-suoyin-202and0.75.sql")
    read_idealab_and_concat_final_data(original_table_name=original_table_datas, write_table_name=write_table_name, idealab_table_name=jianfei_table, model_name=model_name, use_old_extra_id=True, risk_type="jianfei")

def concat_deepseekvl2_image(original_table_name:str):
    original_datas = read_data_from_odps(original_table_name)
    write_table_name = f"{whole_pt_name}/model=deepseekvl2"
    with open("deepseekvl2_datas.json", 'r', encoding='utf-8') as f:
        whole_datas=json.load(f)
    id_answer_dict = {}
    dataset_content_set = set()
    whole_datas_new = []
    repeate_count = 0
    for item in whole_datas:
        if item['content'] in dataset_content_set:
            repeate_count += 1
            print(f"NO.{repeate_count}, has repeate")
        else:
            id_answer_dict[str(item['id'])] = item
            whole_datas_new.append(item)

    final_datas = []
    hasnot_key_count = 0
    id_content_not_match_count = 0
    for index, item in original_datas.iterrows():
        new_item = {}
        for key, value in item.items():
            new_item[key] = value
        if str(new_item['id']) not in id_answer_dict.keys():
            hasnot_key_count += 1
            print(f"NO.{hasnot_key_count}, has not key")
        else:
            answer_item = id_answer_dict[str(new_item['id'])]
            if new_item['content'] != answer_item['content']:
                id_content_not_match_count += 1
                print(f"NO.{id_content_not_match_count}, id not match cotnent.")
            new_item['分析'] = answer_item['分析']
            new_item['关键词'] = answer_item['关键词']
            new_item['选项'] = answer_item['选项']
            final_datas.append(new_item)

    write_train_data_to_table(final_datas, write_table_name)

def handle_image_from_six_risks():
    original_table_name = f"{whole_pt_name}/model=human"
    # concat_human_image(original_table_name)
    # concat_llava16_image(original_table_name)
    # concat_minicpmv_image(original_table_name)
    # concat_internvl8b_image(original_table_name)
    # concat_internvl14b_image(original_table_name)
    # concat_internvl38b_image(original_table_name)
    # concat_qwenvl7b_image(original_table_name)
    # concat_qwenvl32b_image(original_table_name)
    # concat_qwenvl72b_image(original_table_name)
    # concat_qwenvlmax_image(original_table_name)
    # concat_deepseekvl2_image(original_table_name)

    # human_data = get_human_table_datas_to_list(original_table_name)
    # concat_gpt4o_image(original_table_name, human_data)
    # concat_claude37_image(original_table_name, human_data)
    # concat_gemini25pro_image(original_table_name, human_data)

if __name__ == '__main__':
    handle_image_from_six_risks()

    # batch_3b3e7156-f94f-463c-bf78-9ef09faa59c0
    # odps_image_main(table=f"{whole_pt_name}/model=human", model_name='qwen-vl-max')
