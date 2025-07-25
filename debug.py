from model import load_tokenizer
from utils import load_and_prepare_train_and_val_data

def main():
    tokenizer = load_tokenizer("granite")
    train_dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc"]
    train_dataset, val_dataset = load_and_prepare_train_and_val_data(tokenizer, train_dataset_shortcodes)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
if __name__ == "__main__":
    main()