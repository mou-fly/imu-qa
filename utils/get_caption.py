import numpy as np
import pandas as pd
from fontTools.designspaceLib import posix
from tqdm import tqdm

# from models.m3 import imu_input
# from test.data_p import h5_file
from test.read_data import root_path
from utils import h5
import glob
import os
from utils.h5 import load_h5, save_h5_and_csv
from openai import OpenAI
from scipy.interpolate import interp1d

action_to_id = {
    '伸懒腰': 0, '给杯子倒水': 1, '写字': 2, '切水果': 3, '吃水果': 4, '吃药': 5,
    '喝水': 6, '坐下': 7, '开关护眼灯': 8, '开关窗帘': 9, '开关窗户': 10, '打字': 11,
    '打开信封': 12, '扔垃圾': 13, '拿水果': 14, '捡东西': 15, '接电话': 16,
    '操作鼠标': 17, '擦桌子': 18, '板书': 19, '洗手': 20, '玩手机': 21,
    '看书': 22, '给植物浇水': 23, '走向床': 24, '走向椅子': 25, '走向橱柜': 26,
    '走向窗户': 27, '走向黑板': 28, '起床': 29, '起立': 30, '躺下': 31,
    '静止站立': 32, '静止躺着': 33
}

id_to_action = {
    0: '伸懒腰', 1: '给杯子倒水', 2: '写字', 3: '切水果', 4: '吃水果', 5: '吃药',
    6: '喝水', 7: '坐下', 8: '开关护眼灯', 9: '开关窗帘', 10: '开关窗户', 11: '打字',
    12: '打开信封', 13: '扔垃圾', 14: '拿水果', 15: '捡东西', 16: '接电话',
    17: '操作鼠标', 18: '擦桌子', 19: '板书', 20: '洗手', 21: '玩手机',
    22: '看书', 23: '给植物浇水', 24: '走向床', 25: '走向椅子', 26: '走向橱柜',
    27: '走向窗户', 28: '走向黑板', 29: '起床', 30: '起立', 31: '躺下',
    32: '静止站立', 33: '静止躺着'
}

# Translation mapping
# translations = {
#     '伸懒腰': 'Stretching', '给杯子倒水': 'Pour water into the cup', '写字': 'Writing', '切水果': 'Cutting Fruit',
#     '吃水果': 'Eating Fruit', '吃药': 'Taking Medicine', '喝水': 'Drinking Water', '坐下': 'Sitting Down',
#     '开关护眼灯': 'Turning On/Off Eye Protection Lamp', '开关窗帘': 'Opening/Closing Curtains',
#     '开关窗户': 'Opening/Closing Windows', '打字': 'Typing', '打开信封': 'Opening Envelope',
#     '扔垃圾': 'Throwing Garbage', '拿水果': 'Picking Fruit', '捡东西': 'Picking Up Items', '接电话': 'Answering Phone',
#     '操作鼠标': 'Using Mouse', '擦桌子': 'Wiping Table', '板书': 'Writing on Blackboard', '洗手': 'Washing Hands',
#     '玩手机': 'Using Phone', '看书': 'Reading', '给植物浇水': 'Watering Plants', '走向床': 'Walking to Bed',
#     '走向椅子': 'Walking to Chair', '走向橱柜': 'Walking to Cabinet', '走向窗户': 'Walking to Window',
#     '走向黑板': 'Walking to Blackboard', '起床': 'Getting Out of Bed', '起立': 'Standing Up',
#     '躺下': 'Lying Down', '静止站立': 'Standing Still', '静止躺着': 'Lying Still'
# }
translations = {
    '伸懒腰': 'stretches',
    '给杯子倒水': 'pours water into the cup',
    '写字': 'writes',
    '切水果': 'cuts fruit',
    '吃水果': 'eats fruit',
    '吃药': 'takes medicine',
    '喝水': 'drinks water',
    '坐下': 'sits down',
    '开关护眼灯': 'turns on / off the eye protection lamp',
    '开关窗帘': 'opens / closes the curtains',
    '开关窗户': 'opens / closes the window',
    '打字': 'types',
    '打开信封': 'opens an envelope',
    '扔垃圾': 'throws garbage away',
    '拿水果': 'picks fruit',
    '捡东西': 'picks up an item',
    '接电话': 'answers the phone',
    '操作鼠标': 'uses the mouse',
    '擦桌子': 'wipes the table',
    '板书': 'writes',
    '洗手': 'washes hands',
    '玩手机': 'uses the phone',
    '看书': 'reads',
    '给植物浇水': 'waters the plants',
    '走向床': 'walks to the bed',
    '走向椅子': 'walks to the chair',
    '走向橱柜': 'walks to the cabinet',
    '走向窗户': 'walks to the window',
    '走向黑板': 'walks to the blackboard',
    '起床': 'gets out of bed',
    '起立': 'stands up',
    '躺下': 'lies down',
    '静止站立': 'stands still',
    '静止躺着': 'lies still',
}


# 遍历并处理每个文件
def caption_by_deepseek(file, client, new_file):
    data = file["data"]
    label = file["label"]
    # 一个动作序列包含3个动作
    action_num = 3
    action_sequence_num = len(label) - action_num + 1
    for i in range(action_sequence_num):
        start_frame = label[i][2]
        end_frame = label[i + action_num - 1][3]
        window_sequence = data[:, start_frame:end_frame, :]
        target_length = 1500
        current_length = window_sequence.shape[1]
        pad_width = target_length - current_length
        if pad_width > 0:
            # pad_width 参数：(前补, 后补) 的元组，对每个维度指定
            window_sequence_padded = np.pad(
                window_sequence,
                pad_width=((0, 0), (0, pad_width), (0, 0)),  # 只在第2维后补 0
                mode='constant',
                constant_values=0
            )
        else:
            window_sequence_padded = window_sequence[:, 0:1500, :]  # 如果已达到或超过1500，不补
        # if window_sequence_padded.shape != (5, 1500, 6):
        #     print(window_sequence_padded.shape)
        action_id_sequence = [label[j][1] for j in range(i, i + action_num)]
        action_cn_sequence = [id_to_action[id] for id in action_id_sequence]
        action_en_sequence = [translations[action] for action in action_cn_sequence]
        caption = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": 'I will give you several consecutive actions taken by one person. Please describe this '
                            'person\'s behavior in one sentence. People call it the third perspective (human). Please '
                            'make sure to include every action that appears in the sentence! I will give you several '
                            'examples: 1-A man lights a match book on fire. 2-An elderly man is playing the piano in front of a crowd.'},
                {"role": "user",
                 "content": f"1-{action_en_sequence[0]} 2-{action_en_sequence[1]} 3-{action_en_sequence[2]}"},
            ],
            stream=False
        ).choices[0].message.content
        new_file["data"].append(window_sequence_padded)
        new_file["label"].append(
            [start_frame, end_frame, action_id_sequence, action_en_sequence, action_cn_sequence, caption])
        # new_file["caption"].append(caption)

        # print("i:", i)
        # print("action_id_sequence:", action_id_sequence)
        # print("action_cn_sequence:", action_cn_sequence)
        # print("action_en_sequence:", action_en_sequence)
        # print("start_frame:", start_frame)
        # print("end_frame:", end_frame)
        # print("window_sequence:", window_sequence)

def caption_by_concat(file, new_file, num_exceed=None, num_all=None):
    data = file["data"]
    label = file["label"]
    # 一个动作序列包含3个动作
    action_num = 5
    action_sequence_num = len(label) - action_num + 1
    for i in range(action_sequence_num):
        num_all += 1
        start_frame = label[i][2]
        end_frame = label[i + action_num - 1][3]
        window_sequence = data[:, start_frame:end_frame, :]
        target_length = 2500
        current_length = window_sequence.shape[1]
        pad_width = target_length - current_length
        if pad_width > 0:
            # pad_width 参数：(前补, 后补) 的元组，对每个维度指定
            window_sequence_padded = np.pad(
                window_sequence,
                pad_width=((0, 0), (0, pad_width), (0, 0)),  # 只在第2维后补 0
                mode='constant',
                constant_values=0
            )
        else:
            window_sequence_padded = window_sequence[:, 0:2500, :]  # 如果已达到或超过1500，不补
            if pad_width < 0:
                num_exceed += 1

        action_id_sequence = [label[j][1] for j in range(i, i + action_num)]
        action_cn_sequence = [id_to_action[id] for id in action_id_sequence]
        action_en_sequence = [translations[action] for action in action_cn_sequence]

        caption = "A person {} , {} , {} , {} and {} .".format(action_en_sequence[0], action_en_sequence[1],
                                                               action_en_sequence[2], action_en_sequence[3],
                                                               action_en_sequence[4])
        new_file["data"].append(window_sequence_padded)
        new_file["label"].append(
            [start_frame, end_frame, action_id_sequence, action_en_sequence, action_cn_sequence, caption])

    return num_exceed, num_all

def upsample(data, target_length):
    original_len = data.shape[0]
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, data, axis=0, kind='linear')
    return f(x_new)

def caption(file, new_file, fps=50, win_sec=30, stride_sec=5, min_ratio=0.8, pods_file=None):
    data = file["data"]  # shape: [5, 2500, 6]
    labels = file["label"]  # shape: [N, 4] -> [id, class, start, end]
    if pods_file:
        pods = pods_file["data"][:, 3:]
    pods = upsample(pods, data.shape[1])
    pods = np.expand_dims(pods, axis=0)
    data = np.concatenate([data, pods], axis=0)

    num_device, n_frame, feat_dim = data.shape
    win_len = win_sec * fps
    stride = stride_sec * fps
    num_seq = (n_frame - win_len) // stride + 1

    for i in range(num_seq):
        index_start = i * stride
        index_end = index_start + win_len
        window = data[:, index_start:index_end, :]

        window_labels = []

        for label in labels:
            action_class = int(label[1])
            act_start = int(label[2])
            act_end = int(label[3])
            act_len = act_end - act_start

            # 如果动作开始都在窗口之后，后面的动作也都不可能匹配，直接退出循环
            if act_start > index_end:
                break

            # 如果动作完全在窗口前，跳过
            if act_end < index_start:
                continue

            # 计算与当前窗口的交集
            overlap_start = max(index_start, act_start)
            overlap_end = min(index_end, act_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap >= min_ratio * act_len:
                window_labels.append(action_class)

        sentence = label_ids_to_sentence(window_labels, id_to_action, translations)
        new_file["data"].append(window)
        new_file["label"].append(sentence)


def caption_with_padding_and_distribution(file, new_file, fps=50, win_sec=30, stride_sec=0.2):
    data = file["data"]  # [num_device, n_frame, feat_dim]
    labels = file["label"]  # [[id, class, start, end], ...]

    num_device, n_frame, feat_dim = data.shape
    win_len = int(win_sec * fps)
    stride = int(stride_sec * fps)

    action_sequence_num = 0
    for label in labels:
        action_class = int(label[1])
        act_start = int(label[2])
        act_end = int(label[3])
        if act_start > (n_frame - win_len):
            break
        else:
            action_sequence_num += 1

    for i in range(action_sequence_num):
        start_frame = labels[i][2]
        end_frame = start_frame + win_len
        window_label = []
        window_data = data[:, start_frame:end_frame, :]
        for j in range(i, len(labels)):
            label = labels[j]
            action_class = int(label[1])
            act_start = int(label[2])
            act_end = int(label[3])
            if act_end > end_frame:
                break
            else:
                window_label.append(action_class)
        sentence = label_ids_to_sentence(window_label, id_to_action, translations)
        new_file["data"].append(window_data)
        new_file["label"].append(sentence)
        if i == 0:
            new_file["data"].append(data[:, start_frame + stride:end_frame + stride, :])
        else:
            new_file["data"].append(data[:, start_frame - stride:end_frame - stride, :])
        new_file["label"].append(sentence)


def label_ids_to_sentence(label_ids, id_to_action, translations):
    if not label_ids:
        return None
    # 去重并保持顺序
    unique_ids = []
    for id in label_ids:
        if id not in unique_ids:
            unique_ids.append(id)

    # 中文动作列表
    action_cn_sequence = [id_to_action[id] for id in unique_ids]
    # 英文动作列表
    action_en_sequence = [translations[action] for action in action_cn_sequence]

    # 拼接自然语言
    if len(action_en_sequence) == 1:
        sentence = f"A person {action_en_sequence[0]}."
    else:
        sentence = f"A person {' , '.join(action_en_sequence[:-1])} , and {action_en_sequence[-1]} ."

    return sentence


if __name__ == "__main__":
    # 指定目录路径
    test_scene = 3
    directory1 = r"C:\project\caption\data\WWADL_open\imu"
    directory2 = r"C:\project\caption\data\WWADL_open\AirPodsPro"
    root_path = r"C:\project\caption\data\processed_data_joint_30s_test_{}".format(test_scene)
    output_h5_test = os.path.join(root_path, "data_test.h5")
    output_csv_test = os.path.join(root_path, "label_test.csv")
    output_h5_train = os.path.join(root_path, "data_train.h5")
    output_csv_train = os.path.join(root_path, "label_train.csv")
    # output_h5_test = r"C:\project\caption\data\processed_data_joint\data_test.h5"
    # output_csv_test = r"C:\project\caption\data\processed_data_joint\label_test.csv"
    # output_h5_train = r"C:\project\caption\data\processed_data_joint\data_train.h5"
    # output_csv_train = r"C:\project\caption\data\processed_data_joint\label_train.csv"
    # h5_files = glob.glob(os.path.join(directory, "*.h5"))
    # print(h5_files)
    processed_results = {}
    client = OpenAI(api_key="sk-7f6cc357db9948ffb73a7a3b72ddf20a", base_url="https://api.deepseek.com")


    train_h5_files = []
    train_h5_files_ = []
    train_path = pd.read_csv("C:\project\caption\data\imu_train.csv")
    for index, row in train_path.iterrows():
        scene = row.scene_id
        if scene != test_scene:
            name = row.file_name
            p = os.path.join(directory1, name)
            q = os.path.join(directory2, name)
            train_h5_files.append(p)
            train_h5_files_.append(q)


    test_h5_files = []
    test_h5_files_ = []
    test_path = pd.read_csv("C:\project\caption\data\imu_test.csv")
    for index, row in test_path.iterrows():
        scene = row.scene_id
        if scene == test_scene:
            name = row.file_name
            p = os.path.join(directory1, name)
            q = os.path.join(directory2, name)
            test_h5_files.append(p)
            test_h5_files_.append(q)


    # train_names = train_path["file_name"].tolist()
    # train_h5_files = []
    # train_h5_files_ = []
    # for name in train_names:
    #     p = os.path.join(directory1, name)
    #     q = os.path.join(directory2, name)
    #     train_h5_files.append(p)
    #     train_h5_files_.append(q)
    # test_path = pd.read_csv("C:\project\caption\data\imu_test.csv")
    # test_names = test_path["file_name"].tolist()
    # test_h5_files = []
    # test_h5_files_ = []
    # for name in test_names:
    #     p = os.path.join(directory1, name)
    #     q = os.path.join(directory2, name)
    #     test_h5_files.append(p)
    #     test_h5_files_.append(q)

    train_files = {
        "data": [],
        "label": []
    }
    test_files = {
        "data": [],
        "label": []
    }
    num_exceed = 0
    num_all = 0
    cnt = 0
    for i in range(len(train_h5_files)):
        p = train_h5_files[i]
        q = train_h5_files_[i]
        # if cnt == 100:
        #     break
        cnt += 1
        h5_data = load_h5(p)
        pods = load_h5(q)
        # num_exceed, num_all = caption_by_concat(h5_data, train_files, num_exceed, num_all)
        caption(h5_data, train_files, pods_file=pods)
    save_h5_and_csv(output_h5_train, output_csv_train, train_files)
    #
    for i in range(len(test_h5_files)):
        p = test_h5_files[i]
        q = test_h5_files_[i]
        # if cnt == 150:
        #     break
        cnt += 1
        h5_data = load_h5(p)
        pods = load_h5(q)
        # num_exceed, num_all = caption_by_concat(h5_data, test_files, num_exceed, num_all)
        caption(h5_data, test_files, pods_file=pods)

    save_h5_and_csv(output_h5_test, output_csv_test, test_files)

    print("num_exceed : ", num_exceed)
    print("num_all :  ", num_all)
