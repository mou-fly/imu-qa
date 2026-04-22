import os

from data_provider.data_loader import Dataset_raw, Dataset_img
from torch.utils.data import DataLoader

data_dict = {
    'raw': Dataset_raw,
    'img': Dataset_img
}

def data_provider(args, flag):
    Data = data_dict[args.modality]
    if flag == 'test':
        data_path = args.test_data_path if args.modality=='raw' else "pic_test.h5"
        label_path = args.test_label_path
    else:
        data_path = args.train_data_path if args.modality=='raw' else "pic_train.h5"
        label_path = args.train_label_path


    data_set = Data(
        root_path = args.root_path,
        data_path = data_path,
        flag = flag,
        use_patch = args.use_patch,
        label_path = label_path,
        caption_len = args.caption_max_len,
        patch_len = args.patch_len,
        stride = args.stride,
        normalize = getattr(args, "normalize", False),
        norm_mode = getattr(args, "norm_mode", "device_channel"),
        norm_eps = getattr(args, "norm_eps", 1e-6),
        norm_ref_data_path = getattr(args, "norm_ref_data_path", None),
    )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    return data_set, data_loader
