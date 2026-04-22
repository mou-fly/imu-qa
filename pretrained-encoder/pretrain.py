from pretrain_config import build_args
from pretrain_engine import train_pretrain


if __name__ == "__main__":
    args = build_args()
    train_pretrain(args)
