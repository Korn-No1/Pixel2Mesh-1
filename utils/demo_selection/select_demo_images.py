import json
import os
import random
import shutil

#大概率是给demo使用的
#作用：test_tf中每一类随机选择一张图片，复制到example文件夹中

#打开shapenet.json：标签说明
#labels_map是一个dic
#{"04256520": {"id": "04256520","name": "sofa,couch,lounge"}} 一个关于dic的dic
with open("datasets/data/shapenet/meta/shapenet.json") as fp:
    labels_map = json.load(fp)

#打开test_tf：以供test的文件路径汇总
with open("datasets/data/shapenet/meta/test_tf.txt") as fp:
    #line.strip():remove leading and trailing whitespace
    lines = [line.strip() for line in fp.readlines()]

for entry in labels_map.values():
    #entry: e.g. {'id': '04256520', 'name': 'sofa,couch,lounge'}
    #这一步把lines中有04256420的行全挑出来
    file_list = list(filter(lambda x: (entry["id"] + "/") in x, lines))
    #随机选择一个file
    chosen = random.choice(file_list)
    #找到图片的path
    file_location = os.path.join("datasets/data/shapenet/data_tf",
                                 chosen[len("Data/ShapeNetP2M/"):-4] + ".png")
    #复制这张图片到examples文件夹
    shutil.copyfile(file_location, "datasets/examples/%s.png" % entry["name"].split(",")[0])
