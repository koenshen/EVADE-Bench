import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def download_and_process_image(image_dict: Dict[str, str], save_dir: str = "temp_images") -> tuple:
    """
    下载并处理图片，返回特征
    """
    try:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        image_id = image_dict['id']
        image_url = image_dict['content']
        save_path = os.path.join(save_dir, f"{image_id}.jpg")

        # 如果图片已经下载过，直接读取
        if os.path.exists(save_path):
            img = Image.open(save_path)
        else:
            # 下载图片
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content))
            # 保存图片
            img.save(save_path)

        # 图片预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0)
        return image_dict, img_tensor

    except Exception as e:
        print(f"Error processing image {image_dict['id']}: {str(e)}")
        return image_dict, None


def dataset_filtering_pipeline(
        text_list: List[Dict[str, str]],
        image_list: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    数据集过滤流程
    """

    # Stage 1: ID Deduplication
    def unique_by_id(data_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen_ids = set()
        unique_items = []
        for item in data_list:
            if item['id'] not in seen_ids:
                seen_ids.add(item['id'])
                unique_items.append(item)
        return unique_items

    print("Stage 1: Performing ID deduplication...")
    D_img = unique_by_id(image_list)
    D_txt = unique_by_id(text_list)

    # Stage 2: Clustering & Sampling
    def get_features(data_list: List[Dict[str, str]], modality: str):
        if modality == 'txt':
            # 文本特征提取
            vectorizer = TfidfVectorizer()
            contents = [item['content'] for item in data_list]
            return vectorizer.fit_transform(contents).toarray()
        else:
            # 图片特征提取
            print("Downloading and processing images...")
            # 使用预训练模型提取特征
            feature_extractor = models.resnet18(pretrained=True)
            feature_extractor.eval()

            # 并行下载和处理图片
            features = []
            valid_items = []

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(download_and_process_image, item)
                           for item in tqdm(data_list)]

                for future in tqdm(futures):
                    item, img_tensor = future.result()
                    if img_tensor is not None:
                        with torch.no_grad():
                            feature = feature_extractor(img_tensor)
                            features.append(feature.numpy().flatten())
                            valid_items.append(item)

            return np.array(features), valid_items

    def cluster_and_sample(
            data_list: List[Dict[str, str]],
            modality: str,
            n_clusters: int = 300,
            samples_per_cluster: int = 60
    ) -> List[Dict[str, str]]:
        if len(data_list) < n_clusters:
            n_clusters = max(len(data_list) // 2, 1)

        print(f"Extracting features for {modality} data...")
        if modality == 'img':
            features, valid_items = get_features(data_list, modality)
            data_list = valid_items  # 更新为只包含有效图片的列表
        else:
            features = get_features(data_list, modality)

        print(f"Clustering {modality} data...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)

        # 按簇组织数据
        cluster_data = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_data[cluster_id].append(data_list[idx])

        # 从每个簇中随机采样
        D_sampled = []
        for cluster in cluster_data.values():
            sample_size = min(samples_per_cluster, len(cluster))
            D_sampled.extend(random.sample(cluster, sample_size))

        return D_sampled

    print("Stage 2: Performing clustering and sampling...")
    D_balanced = {
        'img': cluster_and_sample(D_img, 'img'),
        'txt': cluster_and_sample(D_txt, 'txt')
    }

    # Stage 3: Model Validation
    def has_disagreement(predictions: List[bool]) -> bool:
        return len(set(predictions)) > 1

    def simulate_model_predictions(sample: Dict[str, str]) -> List[bool]:
        return [random.choice([True, False]) for _ in range(3)]

    print("Stage 3: Performing model validation...")
    D_unlabeled_txt = []
    D_unlabeled_img = []

    for sample in tqdm(D_balanced['txt'], desc="Processing texts"):
        predictions = simulate_model_predictions(sample)
        if has_disagreement(predictions):
            D_unlabeled_txt.append(sample)

    for sample in tqdm(D_balanced['img'], desc="Processing images"):
        predictions = simulate_model_predictions(sample)
        if has_disagreement(predictions):
            D_unlabeled_img.append(sample)

    # 清理临时图片文件
    print("Cleaning up temporary files...")
    import shutil
    if os.path.exists("temp_images"):
        shutil.rmtree("temp_images")

    return D_unlabeled_txt, D_unlabeled_img


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    example_text_list = [
        {"id": f"txt_{i}", "content": f"This is text content {i}"}
        for i in range(100)
    ]
    example_image_list = [
        {"id": f"img_{i}", "content": f"http://example.com/image_{i}.jpg"}
        for i in range(100)
    ]

    # 运行过滤流程
    filtered_texts, filtered_images = dataset_filtering_pipeline(
        example_text_list,
        example_image_list
    )

    # 打印结果
    print(f"原始文本数量: {len(example_text_list)}")
    print(f"过滤后文本数量: {len(filtered_texts)}")
    print(f"原始图片数量: {len(example_image_list)}")
    print(f"过滤后图片数量: {len(filtered_images)}")
