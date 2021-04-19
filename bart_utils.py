import os

from torch.utils.data import Dataset
from tqdm import tqdm


class SummarizationDataset(Dataset):
    label_sep = '__label__'

    def __init__(self, tokenizer, data_dir="./cnn-dailymail/cnn_dm/", type_path="train", block_size=1024):
        super(SummarizationDataset,).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []

        self.source_style = []

        # count = 0
        with open(os.path.join(data_dir, type_path + ".source"), "r") as f:
            for line in tqdm(f.readlines(), desc=f"Loading {type_path}.source"):  # each text is a line and a full story
                text, text_style = line.strip('\"|\n').split(SummarizationDataset.label_sep)
                self.source_style.append(int(text_style))
                tokenized = tokenizer.batch_encode_plus(
                    [text],
                    max_length=block_size,
                    truncation=True,
                    padding='max_length',
                    return_tensors="pt"
                )
                self.source.append(tokenized)

                # count += 1
                # if count > 100:
                #     print("Stopping data loading!!!")
                #     break

        # count = 0
        with open(os.path.join(data_dir, type_path + ".target"), "r") as f:
            for text in tqdm(f.readlines(), desc=f"Loading {type_path}.target"):  # each text is a line and a summary
                text=text.strip('\"|\n')
                tokenized = tokenizer.batch_encode_plus(
                    [text],
                    max_length=56,
                    truncation=True,
                    padding='max_length',
                    return_tensors="pt"
                )
                self.target.append(tokenized)

                # count += 1
                # if count > 100:
                #     print("Stopping data loading!!!")
                #     break

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        text_style = self.source_style[index]
        src_mask = self.source[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "source_style": text_style ,"target_ids": target_ids}