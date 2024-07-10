'''
Refer to this forum post for why it is a good practice to calculate the mean and standard deviation of 
your own dataset (as opposed to using ImageNet's): https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670
'''

import torch
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from code.cnn.dataloaders import MemesDataset
from tqdm import tqdm
from code.cnn.dataloaders import my_collate_fn
import os
from argparse import ArgumentParser

class OnlineMeanStd:
    def __init__(self):
        pass

    def __call__(self, dataset, batch_size, method='strong'):
        """
        Calculate mean and std of a dataset in lazy mode (online)
        On mode strong, batch size will be discarded because we use batch_size=1 to minimize leaps.

        :param dataset: Dataset object corresponding to your dataset
        :param batch_size: higher size, more accurate approximation
        :param method: weak: fast but less accurate, strong: slow but very accurate - recommended = strong
        :return: A tuple of (mean, std) with size of (3,)
        """

        if method == 'weak':
            loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=0)
            mean = 0.
            std = 0.
            nb_samples = 0.
            for X,y in loader:
                data = X
                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                mean += data.mean(2).sum(0)
                std += data.std(2).sum(0)
                nb_samples += batch_samples

            mean /= nb_samples
            std /= nb_samples

            return mean, std

        elif method == 'strong':
            loader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=0,
                                collate_fn=my_collate_fn)
            cnt = 0
            fst_moment = torch.empty(3)
            snd_moment = torch.empty(3)
            print(len(loader), "number of batches")
            for X,y in tqdm(loader, total=len(loader)):
                try:
                    b, c, h, w = X.shape
                    if c == 4:
                        # we omit the alpha channel
                        X = X[:, :3, :, :]
                    nb_pixels = b * h * w
                    sum_ = torch.sum(X, dim=[0, 2, 3])
                    sum_of_square = torch.sum(X ** 2, dim=[0, 2, 3])
                    fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
                    snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
                    cnt += nb_pixels
                except:
                    print("Error in batch")
                    continue
            return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-df","--data_df_path", type=str, default="/home/hsdslab/murgi/meme-research-2024/data/processed/gpu_server_meme_entries.parquet")

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    # Example usage:
    print(os.getcwd())
    dataset = MemesDataset(data_df_path=args.data_df_path)
    mean, std = OnlineMeanStd()(dataset,batch_size=1, method='strong')
    with open("mean_std.txt", "w") as f:
        f.write(f"Mean: {mean}\nStd: {std}")
