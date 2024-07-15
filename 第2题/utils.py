import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random


### 数据集
class Dataset:
    @staticmethod
    def unpickle(file):
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    @staticmethod
    def getlabel(n):
        return Dataset.label_names[n].decode("utf-8")

    def __init__(self, *path):
        a = self.unpickle(path[0])
        self.x = a[b"data"].reshape((-1, 3, 32, 32))
        self.y = a[b"labels"]
        for i in range(1, len(path)):
            a = self.unpickle(path[i])
            self.x = np.concatenate((self.x, a[b"data"].reshape((-1, 3, 32, 32))))
            self.y = np.concatenate((self.y, a[b"labels"]))
        self.y = np.array(self.y, dtype="int64")
        self.len = self.y.shape[0]

    def getdata(self, index):
        return self.x[index] / 255, self.y[index]

    def getdatas(self):
        return self.x / 255, self.y

    def getimg(self, index):
        return self.x[index].transpose((1, 2, 0)), self.y[index]

    def showimg(self, index):
        img, label = self.getimg(index)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        return img, label

    def shuffle(self):
        r = np.random.permutation(self.len)
        x = self.x.copy()
        y = self.y.copy()
        for i in range(self.len):
            self.x[i] = x[r[i]]
            self.y[i] = y[r[i]]

    def enhance(self):
        self.x = np.concatenate((self.x, self.x))
        self.y = np.concatenate((self.y, self.y))
        for i in range(self.len):
            self.x[i] = self.x[i, :, :, ::-1]
        self.len *= 2
        self.shuffle()


Dataset.label_names = Dataset.unpickle("data/cifar-10-batches-py/batches.meta")[
    b"label_names"
]

if __name__ == "__main__":
    import tarfile
    with tarfile.open("data/cifar-10-python.tar.gz","r") as f:
        f.extractall("data/")