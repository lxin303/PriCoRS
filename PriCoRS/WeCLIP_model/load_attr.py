import os
import torch
import torch.nn as nn
import WeCLIP.clip as clip
import numpy as np
import json
from tqdm import tqdm


def attr_clustering(dataset_name='isaid', encoder=None, num_atrr_clusters=112,
                    json_file='/leonardo_work/IscrC_CropAI/RS/WeCLIP/attributes_text/gpt4.0_cluster_isaid.json'):

    with open(json_file, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)

    num_classes = len(tuple(descriptions.keys()))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = encoder

    all_descriptions_embeddings = []
    for class_name, desc in tqdm(descriptions.items()):
        sentences = [item.lower() for item in desc]
        encoded_input = clip.tokenize(sentences).to(device)

        with torch.no_grad():
            sentence_embeddings = model.encode_text(encoded_input)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu()
            all_descriptions_embeddings.append(sentence_embeddings)

    all_descriptions_embeddings_tensor = torch.cat(all_descriptions_embeddings, dim=0)

    # 新增：每个类别的 description center
    class_desc_centers = []
    for class_desc in all_descriptions_embeddings:
        center = class_desc.mean(dim=0, keepdim=True)  # [1, D]
        center = torch.nn.functional.normalize(center, p=2, dim=1)
        class_desc_centers.append(center[0])
    class_desc_centers = torch.stack(class_desc_centers, dim=0)  # [K, D]

    # Choose proper cluster number for different datasets. We set 256 for ADE20K.
    n_clusters = num_atrr_clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_descriptions_embeddings_tensor.numpy())

    desc_class_idx_list = []
    for class_idx, class_desc in enumerate(all_descriptions_embeddings):
        desc_class_idx_list.append(torch.tensor([class_idx] * len(class_desc)))
    desc_class_idx_list = torch.cat(desc_class_idx_list)

    ground_truth_all_classes = []
    activated_clusters_all_classes = []
    for class_idx in range(num_classes):
        class_cluster_info = kmeans.labels_[desc_class_idx_list == class_idx]
        activated_clusters = np.unique(class_cluster_info)
        print(activated_clusters)
        activated_clusters_all_classes.append(activated_clusters)

        ground_truth = np.zeros(n_clusters)
        ground_truth[activated_clusters] = 1
        ground_truth_all_classes.append(ground_truth)

    unique_label, unique_count = np.unique(np.array(ground_truth_all_classes), axis=0, return_counts=True)
    print(unique_count)
    if not (unique_count == 1).all():
        confused_labels = np.where((np.array(ground_truth_all_classes) == unique_label[unique_count > 1]).all(axis=1))
        for class_idx in confused_labels[0]:
            print('----------------')
            print(tuple(descriptions.keys())[class_idx])
            print(descriptions[tuple(descriptions.keys())[class_idx]])
            print('----------------')

    ground_truth_all_classes = torch.tensor(np.array(ground_truth_all_classes)).float()

    cluster_embedding_bank = kmeans.cluster_centers_.transpose(1, 0)
    cluster_embedding_bank = torch.tensor(cluster_embedding_bank).float()

    cache = {
        'cluster_embedding_bank': cluster_embedding_bank,          # [D, B]
        'ground_truth_all_classes': ground_truth_all_classes,      # [K, B]
        'desc_feats': all_descriptions_embeddings_tensor.float(),  # [N_desc, D]
        'desc_class_ids': desc_class_idx_list.long(),             # [N_desc]
        'class_desc_centers': class_desc_centers.float(),         # [K, D]
        'class_names': list(descriptions.keys())
    }

    save_path = f'/leonardo_work/IscrC_CropAI/RS/WeCLIP/attributes_text/{dataset_name}_desc_gpt4.0_cluster_{n_clusters}_embedding_bank.pth'
    torch.save(cache, save_path)

    print(f'{dataset_name}_desc_attr_cluster_{n_clusters} loaded.')
    return cache


def attr_aggregate(text_features, model=None, dataset_name='isaid', num_atrr_clusters=112, json_file='/leonardo_work/IscrC_CropAI/RS/WeCLIP/attributes_text/gpt4.0_cluster_isaid.json',topK=0.9):

    attr_pth_file = f'/leonardo_work/IscrC_CropAI/RS/WeCLIP/attributes_text/{dataset_name}_desc_gpt4.0_cluster_{num_atrr_clusters}_embedding_bank.pth'
    if not os.path.exists(attr_pth_file):
        cache = attr_clustering(dataset_name, encoder=model, num_atrr_clusters=num_atrr_clusters, json_file=json_file)
    else:
        cache = torch.load(attr_pth_file)
    
    attr_features = cache['cluster_embedding_bank']
    attr_flag = cache['ground_truth_all_classes']

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # move + cast: do the dot-product in fp32 for stability
    attr_flag = attr_flag.to(device=device)  # keep its dtype (probably int/bool)
    fg_text = text_features.to(device=device)
    attr_features = attr_features.to(device=device, dtype=torch.float16)
    print('fg_text dtype:', fg_text.dtype, 'attr_features dtype:', attr_features.dtype)
    corr = (fg_text @ attr_features).softmax(dim=-1)

    if topK is not None:
        topk = int((1-topK)* attr_features.shape[1])
        attn_logit = (fg_text @ attr_features)
        corr, idx = torch.sort(attn_logit, dim=-1, descending=True)
        # ### TODO select attr
        # idx_select = idx[:,57]
        # attr_se = torch.gather(attr_features,1,idx_select.unsqueeze(0).expand(512,-1))
        # ###
        corr[:,-topk:] = float('-inf')
        restored_corr = torch.zeros_like(corr)
        restored_corr.scatter_(-1, idx, corr)
        corr = restored_corr
        corr = corr.softmax(dim=-1)

    text_attri_agg = corr @ attr_features.t() + fg_text
    # text_attri_agg = attr_se.permute(1,0)
    # text_attri_agg = attr_features.t() + fg_text
    # text_attri_agg = torch.cat([text_attri_agg,bg_text], dim=0)

    text_attri_agg = (text_attri_agg / text_attri_agg.norm(dim=1, keepdim=True))

    return text_attri_agg, attr_flag