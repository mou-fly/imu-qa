import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

def convert_timeseries_to_images_h5(input_h5_path, output_h5_path, dataset_key='data',
                                     dpi=100, figsize=(1.2,1.2)):
    """
    将 HDF5 文件中的时间序列数据绘制为图像，并保存为新的 HDF5 文件。

    参数：
    - input_h5_path: str，原始 HDF5 文件路径（如含时间序列的 data_train.h5）
    - output_h5_path: str，输出图像 HDF5 文件路径（如 pic_train.h5）
    - dataset_key: str，原始数据中使用的主键（默认 'data'）
    - dpi: int，图像分辨率，影响最终图像尺寸
    - figsize: tuple，图像大小（单位：英寸）

    输出：
    - 在 output_h5_path 创建新的 .h5 文件，保存 shape 为 (样本数, 30, H, W, 3) 的图像数据
    """
    # 加载原始数据
    with h5py.File(input_h5_path, 'r') as input_h5:
        data = input_h5[dataset_key]  # shape: (1000, 5, 1500, 6)
        num_samples, num_devices, seq_len, num_channels = data.shape

        # 图像尺寸
        img_height = int(figsize[1] * dpi)
        img_width = int(figsize[0] * dpi)

        # 创建输出 HDF5 文件
        with h5py.File(output_h5_path, 'w') as output_h5:
            image_dataset = output_h5.create_dataset(
                "images",
                shape=(num_samples, num_devices * num_channels, img_height, img_width, 3),
                dtype='uint8'
            )

            # 遍历样本
            for sample_idx in tqdm(range(num_samples), desc="Processing samples"):
                images_per_sample = []

                for device_idx in range(num_devices):
                    for channel_idx in range(num_channels):
                        ts = data[sample_idx, device_idx, :, channel_idx]

                        # 画图并转为RGB图像
                        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                        ax.plot(ts)
                        ax.axis('off')
                        canvas = FigureCanvas(fig)
                        canvas.draw()
                        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                        img = img.reshape((img_height, img_width, 3))
                        plt.close(fig)

                        images_per_sample.append(img)

                # 写入该样本的30张图像
                image_dataset[sample_idx] = np.array(images_per_sample)

    print(f"图像保存完毕：{output_h5_path}")


def draw(input_h5_path, output_root, csv_file=None, dpi=100, figsize=(2.4,2.4)):
    with h5py.File(input_h5_path, 'r') as input_h5:
        data = input_h5['data']  # shape: (N, 5, 1500, 6)
        num_samples, num_devices, seq_len, num_channels = data.shape

        os.makedirs(output_root, exist_ok=True)
        cnt = 0
        # file_pd = pd.read_csv(csv_file)
        # file_names = file_pd['file_name'].tolist()

        for i in range(num_samples):
            # 创建以 cnt 命名的子目录
            sample_dir = os.path.join(output_root, str(cnt))
            os.makedirs(sample_dir, exist_ok=True)

            for j in range(num_devices):
                plt.figure(figsize=figsize, dpi=dpi)

                for k in range(num_channels):
                    ts = data[i, j, :, k]  # shape: (1500,)
                    plt.plot(range(seq_len), ts)

                plt.tight_layout()

                # 保存图像
                save_path = os.path.join(sample_dir, f'device_{j}.png')
                plt.savefig(save_path)
                plt.close()

            cnt += 1

    print(f"✅ 所有图像已保存到目录：{output_root}")

if __name__ == '__main__':

    draw(
        "C:\project\caption\data\processed_data_joint_30s_small\data_test.h5",
        "C:\project\caption\data\processed_data_joint_30s_small\pic_test",
        # "C:\project\caption\data\processed_data_joint_30s_small\label_test.csv"
    )
    draw(
        "C:\project\caption\data\processed_data_joint_30s_small\data_train.h5",
        "C:\project\caption\data\processed_data_joint_30s_small\pic_train",
        # "C:\project\caption\data\processed_data_joint_30s_small\label_train.csv"
    )

    # with h5py.File("C:\project\caption\data\processed_data_joint_30s_small\pic_train.h5", 'r') as input_h5:
    #     print(input_h5.keys())
    #     data = input_h5["images"]
    # with h5py.File("C:\project\caption\data\processed_data_joint_30s_small\data_train.h5", 'r') as input_h5:
    #     print(input_h5.keys())
    #     data = input_h5["data"]
    # convert_timeseries_to_images_h5(
    #     "C:\project\caption\data\processed_data_joint_30s_small\data_test.h5",
    #     "C:\project\caption\data\processed_data_joint_30s_small\pic_test.h5"
    # )
    # convert_timeseries_to_images_h5(
    #     "C:\project\caption\data\processed_data_joint_30s_small\data_train.h5",
    #     "C:\project\caption\data\processed_data_joint_30s_small\pic_train.h5"
    # )
    #
    #
    # convert_timeseries_to_images_h5(
    #     "C:\project\caption\data\processed_data_joint_30s\data_test.h5",
    #     "C:\project\caption\data\processed_data_joint_30s\pic_test.h5"
    # )
    # convert_timeseries_to_images_h5(
    #     "C:\project\caption\data\processed_data_joint_30s\data_train.h5",
    #     "C:\project\caption\data\processed_data_joint_30s\pic_train.h5"
    # )

