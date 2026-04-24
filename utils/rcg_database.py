from data_provider.data_factory import data_provider
from models.rcg import IMUEncoder, TextEncoder
import tqdm
import torch
import h5py

from utils.h5 import load_h5


def get_rcg_database(args):
    data, loader = data_provider(args, 'val')
    process_file = {
        "data": [],
        "label":[]
    }
    for i, (batch_x, batch_y, mask) in enumerate(loader):
        batch_x = torch.tensor(batch_x, dtype=torch.float32, device=args.device)
        batch_y = torch.tensor(batch_y, dtype=torch.long, device=args.device)
        imu_encoder = IMUEncoder().to(args.device)
        text_encoder = TextEncoder().to(args.device)
        e_imu = imu_encoder(batch_x)
        e_text = text_encoder(batch_y)
        if len(e_imu) == args.batch_size:
            process_file["data"].append(e_imu.cpu().detach().numpy())
            process_file["label"].append(e_text.cpu().detach().numpy())

    with h5py.File("C:\project\caption\data\\rcg_database.h5", 'w') as f:
        f.create_dataset("data", data=process_file["data"])
        f.create_dataset("label", data=process_file["label"])
        f.close()

if __name__ == '__main__':
    file = load_h5("C:\project\caption\data\\rcg_database.h5")
    # data = file["data"][:]
    label = file["label"][:]
    for i in range(1):
        # print(data)
        print(label)


