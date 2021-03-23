from torch.utils.data import DataLoader, IterableDataset
import os
import numpy as np
import cv2
import numpy.random as random


class Videos(IterableDataset):
    def __init__(self, datadir, t=50, batch_size=49):
        self.datadir = datadir
        self.t = t
        self.batch_size = batch_size

    def __iter__(self):
        directory = os.path.abspath(self.datadir)
        cache = {}
        # random = np.random.RandomState(42)
        for episode in os.listdir(directory):
            vid = cv2.VideoCapture(os.path.join(directory, episode, 'video.avi'))
            wid = int(vid.get(3))
            hei = int(vid.get(4))
            framenum = int(vid.get(7))
            npvid = np.zeros((framenum, hei, wid, 3), dtype=np.uint8)
            i = 0
            a = True
            while vid.isOpened() and a:
                a, b = vid.read()
                if a:
                    npvid[i] = b
                i += 1
            vid.release()
            del vid
            data = np.load(os.path.join(directory, episode, 'data.npz'))
            data = {k: data[k] for k in data.keys()}
            data['image'] = npvid
            cache[episode] = data
            keys = list(cache.keys())
        self.cache = list(cache.values())
        self.lens = [len(o['action']) for o in self.cache]
        while True:
            for index in random.choice(len(keys), self.batch_size, replace=False):
                episode = cache[keys[index]]
                if self.t:
                    total = len(next(iter(episode.values())))
                    available = total - self.t
                    if available < 1:
                        continue
                    index = int(random.randint(0, available))
                    episode = {k: v[index: index + self.t] for k, v in episode.items()}
                    episode['image'] = np.transpose(episode['image'].astype(np.float32) / 255., (0, 3, 1, 2))
                    episode['position'] = episode['position'][:, 1:4].astype(np.float32)
                    episode['image'] = episode['image'][..., 16:-16]
                yield episode


def get_loader(datadir, n_jobs=10, t=50, batch_size=16):
    return DataLoader(Videos(datadir, t=t, batch_size=batch_size), batch_size=batch_size, num_workers=n_jobs)
