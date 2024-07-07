import json

def split_json(file_path, part1_path, part2_path, split_index):
    # 读取JSON文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 分割数据
    part1_data = data[:split_index]
    part2_data = data[split_index:]

    # 写入新的JSON文件
    with open(part1_path, 'w') as part1_file:
        json.dump(part1_data, part1_file, indent=4)

    with open(part2_path, 'w') as part2_file:
        json.dump(part2_data, part2_file, indent=4)

# 这里你可以替换为你的JSON文件路径和输出文件名
split_json('case_results_t_n_2017_2.json', 'case_results_t_n_2017_21.json', 'case_results_t_n_2017_22.json', 40)
