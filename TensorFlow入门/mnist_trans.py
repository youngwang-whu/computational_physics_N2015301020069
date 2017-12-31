# 将input目录中的图片经缩小并转化为灰度图片后再存入identify目录
import os
from PIL import Image

def get_path(path):
    return [os.path.join(path, f)for f in os.listdir(path)]

for path in get_path(r'C:\TensorFlow_train\input'):
    outfile = 'C:\TensorFlow_train\identify\ '+ os.path.basename(path)
    print(outfile)
    # 写入图片，并将其转化为灰度图
    im = Image.open(path).convert('L')
    # 缩小图片尺寸到28*28
    im.thumbnail((28, 28))
    im.show()
    im.save(outfile)
