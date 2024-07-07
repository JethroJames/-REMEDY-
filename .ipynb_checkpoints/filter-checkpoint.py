# import json
# import pandas as pd

# # 加载JSON数据
# with open('case_results_t_n_2017.json', 'r') as file:
#     data = json.load(file)

# # 从JSON数据中提取batch_details，并转换为DataFrame
# df = pd.DataFrame(data['batch_details'])

# # 筛选出true_label和pred_label不一致的项
# filtered_df = df[df['true_label'] != df['pred_label']]

# # 获取筛选出的项的索引
# filtered_indices = filtered_df.index.tolist()

# # 从原始数据中获取与batch_details同级的batch数据，这里获取batch_uncertainty中第0个元素的对应批次数据
# batch_data = {
#     'batch_f1_score': data['batch_f1_score'],
#     'batch_classification_report': data['batch_classification_report'],
#     'batch_uncertainty': data['batch_uncertainty'][0]  # 获取第0个元素
# }

# # 提取与filtered_df对应的uncertainty_A的数据
# filtered_uncertainty_A = [data['uncertainty_A'][i] for i in filtered_indices]

# # 提取与filtered_df对应的evidence_A_softmax的数据
# filtered_evidence_A_softmax = [data['evidence_A_softmax'][i] for i in filtered_indices]

# # 提取与filtered_df对应的belief_a的数据
# filtered_belief_a = [data['belief_a'][i] for i in filtered_indices]

# # 将筛选后的DataFrame转换为字典格式
# filtered_batch_details_dict = filtered_df.to_dict(orient='records')

# # 将所有数据保存为JSON文件
# with open('case_results_t_n_2017f.json', 'w') as json_file:
#     json.dump({
#         "batch_data": batch_data,
#         "filtered_batch_details": filtered_batch_details_dict,
#         "filtered_uncertainty_A": filtered_uncertainty_A,
#         "filtered_evidence_A_softmax": filtered_evidence_A_softmax,
#         "filtered_belief_a": filtered_belief_a
#     }, json_file, indent=4)

# print("Data has been saved to 'filtered_data.json'")


import json
import pandas as pd

# 加载JSON数据
with open('case_results_softmax_n_2015.json', 'r') as file:
    data_list = json.load(file)

# 初始化一个空的列表来收集所有的筛选结果
filtered_results = []

# 遍历列表中的每个字典
for data in data_list:
    # 从当前字典中提取batch_details，并转换为DataFrame
    df = pd.DataFrame(data['batch_details'])

    # 筛选出true_label和pred_label不一致的项
    filtered_df = df[df['true_label'] != df['pred_label']]

    # 获取筛选出的项的索引
    filtered_indices = filtered_df.index.tolist()
    filtered_df['original_index'] = filtered_indices
    
    filtered_batch_uncertainty = [
        [
            data['batch_uncertainty'][i][j] for j in filtered_indices
        ] for i in range(len(data['batch_uncertainty']))
    ]


    # 提取与filtered_df对应的数据
    filtered_uncertainty_A = [data['uncertainty_A'][i] for i in filtered_indices]
    filtered_evidence_A_softmax = [data['evidence_A_softmax'][i] for i in filtered_indices]
    filtered_belief_a = [data['belief_a'][i] for i in filtered_indices]

    # 将筛选后的数据保存到filtered_results中
    filtered_results.append({
        "filtered_batch_details": filtered_df.to_dict(orient='records'),
        "batch_uncertainty": filtered_batch_uncertainty,
        "filtered_uncertainty_A": filtered_uncertainty_A,
        "filtered_evidence_A_softmax": filtered_evidence_A_softmax,
        "filtered_belief_a": filtered_belief_a
    })

# 将所有筛选后的数据保存为JSON文件
with open('case_results_softmax_nf_2015.json', 'w') as json_file:
    json.dump(filtered_results, json_file, indent=4)

print("Data has been saved to 'filtered_data.json'")


