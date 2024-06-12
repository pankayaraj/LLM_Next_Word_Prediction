from datasets import load_dataset, DatasetDict
import random
dataset = load_dataset("swj0419/BookMIA", split='train')
#dataset.save_to_disk("/cmlscratch/pan/Watermarking/Datasets/dataset/bookMIA")


dataset_in = dataset.filter(lambda x: x["label"] == 1)
dataset_not_in = dataset.filter(lambda x: x["label"] == 0)



dataset_main = DatasetDict({
    "seen_in_training":dataset_in,
    "not_seen_in_training":dataset_not_in,
})


dataset_main.save_to_disk("/cmlscratch/pan/Watermarking/Datasets/dataset/bookMIA")
