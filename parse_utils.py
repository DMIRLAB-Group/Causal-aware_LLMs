import re
import numpy as np


def parse_causal_relation_with_index(verify_result, correspond_relation, evcg_matrix, evcg_objects, verified_index):

    true_list = []
    false_list = []
    for index, relation in enumerate(correspond_relation):
        cause, effect = relation
        cause_index = evcg_objects[cause]
        effect_index = evcg_objects[effect]

        # 更新verified_index和evcg_matrix
        verified_index.append((cause_index, effect_index))
        evcg_matrix[cause_index][effect_index] = 1 if verify_result[index] else 0

        index_to_entity = {v: k for k, v in evcg_objects.items()}

        if verify_result[index]:
            true_list.append((cause, effect))
        else:
            false_list.append((cause, effect))

    return evcg_matrix, evcg_objects, verified_index, true_list, false_list

def parse_causal_description(description, obs, inv_state):
    invalid_terms = ['null']
    if description is None or len(description) == 0 or description == 'NULL':
        return None
    # 使用正则表达式匹配文本描述中的因果关系
    pattern = re.compile(r'-  (\w+) ->  (\w+)')
    # 找到所有匹配的因果对，并构建从效果到原因的映射
    causal_relationships = pattern.findall(description)
    # 创建一个字典来存储每个对象的因果关系
    causal_dict = {}
    for match in causal_relationships:
        cause, effect = match
        if (cause not in obs and cause not in inv_state['inv'] and cause not in inv_state['status']) or (effect not in obs and effect not in inv_state['inv'] and effect not in inv_state['status']) or cause in invalid_terms or effect in invalid_terms or cause in effect:
            continue
        # 将效果对象添加到字典中，如果没有则初始化为一个集合
        if effect not in causal_dict:
            causal_dict[effect] = set()
        if cause not in causal_dict:
            causal_dict[cause] = set()
        # 添加直接原因
        causal_dict[effect].add(cause)

    return causal_dict


def build_causal_matrix(causal_dict):
    if causal_dict is None or len(causal_dict) == 0 or causal_dict == {}:
        return None, None
    # 确定所有对象的集合
    all_objects = set(causal_dict.keys())
    # 为每个对象创建索引
    object_to_index = {obj: idx for idx, obj in enumerate(all_objects)}

    # 初始化因果矩阵，所有条目都为0
    num_objects = len(object_to_index)
    causal_matrix = [[0] * num_objects for _ in range(num_objects)]

    # 遍历因果关系字典填充矩阵
    for effect, causes in causal_dict.items():
        effect_index = object_to_index[effect]
        for cause in causes:
            cause_index = object_to_index[cause]
            causal_matrix[cause_index][effect_index] = 1

    return causal_matrix, object_to_index


def update_causal_matrix(causal_matrix, object_to_index, new_description, obs, inv):
    # 解析新的描述以获取新的因果关系
    new_causal_dict = parse_causal_description(new_description, obs, inv)
    if new_causal_dict is None or len(new_causal_dict) == 0 or new_causal_dict == {}:
        return causal_matrix, object_to_index
    # 找出新描述中所有涉及的对象，并添加到现有对象集合中
    all_objects = set(object_to_index.keys()) | set(new_causal_dict.keys())
    all_objects |= set(cause for causes in new_causal_dict.values() for cause in causes)
    # 更新对象到索引的映射
    updated_object_to_index = object_to_index.copy()  # 复制现有映射以保留现有索引
    if len(all_objects - set(object_to_index.keys())) == 0:
        return causal_matrix, object_to_index  # 如果没有新对象，直接返回原因果矩阵
    for obj in all_objects - set(object_to_index.keys()):  # 仅对新对象进行索引
        updated_object_to_index[obj] = len(updated_object_to_index)
    # 更新因果矩阵的大小并填充新对象的因果关系
    num_nodes = len(updated_object_to_index)
    updated_causal_matrix = np.array([[0] * num_nodes for _ in range(num_nodes)])

    # 复制现有因果关系到更新的矩阵中
    updated_causal_matrix[:len(causal_matrix), :len(causal_matrix)] = causal_matrix

    # 添加新的因果关系到矩阵中
    for effect, causes in new_causal_dict.items():
        effect_index = updated_object_to_index[effect]
        for cause in causes:
            cause_index = updated_object_to_index[cause]
            updated_causal_matrix[cause_index][effect_index] = 1

    return updated_causal_matrix, updated_object_to_index


def causal_matrix_to_text(causal_matrix, object_to_index):
    if object_to_index is None or len(object_to_index) == 0 or object_to_index == {}:
        return 'null'
    # 初始化一个空列表来保存文本描述的每一行
    descriptions = []
    relation = []

    # 获取对象名称列表，以便可以通过索引访问
    objects = list(object_to_index.keys())

    # 遍历因果矩阵的每一行
    for cause_index, row in enumerate(causal_matrix):
        # 找出所有值为1的列，这些列的索引表示结果对象
        effect_indices = [index for index, value in enumerate(row) if value == 1]

        # 对于每个结果对象，构建描述语句
        for effect_index in effect_indices:
            cause = objects[cause_index]  # 原因对象
            effect = objects[effect_index]  # 结果对象
            description = f"- {cause} -> {effect}."
            if cause != effect:
                descriptions.append(description)
                relation.append((cause,effect))
    # 将描述列表转换为文本描述，描述之间用换行符分隔
    return relation
