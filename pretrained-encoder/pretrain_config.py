import argparse


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("IMU Encoder Pretrain (RQ-VAE + Text Align)")

    # data config (reuse current project data_provider args)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, default="data_train.h5")
    parser.add_argument("--test_data_path", type=str, default="data_test.h5")
    parser.add_argument("--train_label_path", type=str, default="label_train.csv")
    parser.add_argument("--test_label_path", type=str, default="label_test.csv")
    parser.add_argument("--modality", type=str, default="raw")
    parser.add_argument("--caption_max_len", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_patch", type=bool, default=False)
    parser.add_argument("--patch_len", type=int, default=300)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--norm_mode", type=str, default="device_channel", choices=["device_channel", "channel", "global"])
    parser.add_argument("--norm_eps", type=float, default=1e-6)
    parser.add_argument("--norm_ref_data_path", type=str, default="data_train.h5")

    # training config
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--stage", type=str, default="joint", choices=["rqvae", "joint"])
    parser.add_argument("--exp_name", type=str, default="imu_rqvae_align")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint/pretrain")
    parser.add_argument("--log_interval", type=int, default=50)

    # model config
    parser.add_argument("--imu_input_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--imu_encoder_layers", type=int, default=4)
    parser.add_argument("--imu_encoder_heads", type=int, default=8)
    parser.add_argument("--text_vocab_size", type=int, default=1024)
    parser.add_argument("--text_encoder_layers", type=int, default=2)
    parser.add_argument("--text_encoder_heads", type=int, default=8)
    parser.add_argument("--rq_num_embeddings", type=int, default=1024)
    parser.add_argument("--rq_num_quantizers", type=int, default=4)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)

    # loss weights
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_vq", type=float, default=1.0)
    parser.add_argument("--lambda_align", type=float, default=1.0)

    return parser.parse_args()
