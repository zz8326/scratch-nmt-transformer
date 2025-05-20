import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random
import pdb


class CMNTranslationDataset(Dataset):
    def __init__(self, data_path, pretrained = "xlm-roberta-base", max_len = 64):
        super(CMNTranslationDataset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.data = []
        self.max_len = max_len

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                part = line.strip().split('\t')
          
                if len(part) < 2:
                    continue
                src, tgt = part[0], part[1]
                self.data.append((src, tgt))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]

        src = self.tokenizer(
            src,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        tgt = self.tokenizer(
            tgt,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            add_special_tokens=True
        )
        return src["input_ids"].squeeze(0), tgt["input_ids"].squeeze(0)




if __name__ == "__main__":
    dataset = CMNTranslationDataset('./data/cmn.txt')
    src, tgt = dataset[0]

    for i in range(3):  # 測三組看看
        src, tgt = dataset[i]
        print(f"\n--- Sample {i} ---")
        print("SRC:", dataset.tokenizer.decode(src))
        print("TGT:", dataset.tokenizer.decode(tgt))
        print(dataset.tokenizer.pad_token_id)