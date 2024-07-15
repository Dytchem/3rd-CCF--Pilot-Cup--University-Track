import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import random


### 数据增强
# img: (c, h, w)    img_pil: (h, w, c)
def numpy_to_pil(img):
    return Image.fromarray(np.transpose(img, (1, 2, 0)).astype(np.uint8))


def pil_to_numpy(img_pil):
    return np.transpose(np.array(img_pil), (2, 0, 1))


# 随机旋转函数
def random_rotate(img_pil, degrees=10, fillcolor=(128, 128, 128)):
    angle = random.randint(-degrees, degrees)
    return img_pil.rotate(angle, fillcolor=fillcolor)


# 随机噪声函数
def random_noise(img_pil, noise_factor=0.001):
    img_np = np.array(img_pil)
    img_np = img_np.astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_factor * 255, img_np.shape)
    img_np = np.clip(img_np + noise, 0.0, 1.0)

    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


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

    def enhance(self, n=2):
        x = self.x.copy()
        y = self.y.copy()
        for _ in range(n):
            self.x = np.concatenate((x, x, self.x))
            for i in range(self.len):
                img = numpy_to_pil(x[i])
                img1 = random_rotate(img)
                img2 = random_noise(img)
                self.x[i] = pil_to_numpy(img1)
                self.x[i + self.len] = pil_to_numpy(img2)
            self.y = np.concatenate((y, y, self.y))
        self.len = self.len * (2 * n + 1)
        self.shuffle()


Dataset.label_names = Dataset.unpickle("data/cifar-10-batches-py/batches.meta")[
    b"label_names"
]