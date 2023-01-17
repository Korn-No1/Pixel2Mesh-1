import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate

import config
from datasets.base_dataset import BaseDataset


class ShapeNet(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
        super().__init__()
        self.file_root = file_root

        # self.labels_map 读入label并且从零开始排序，以一个dic的形式表示
        with open(os.path.join(self.file_root, "meta", "shapenet.json"), "r") as fp:
            self.labels_map = sorted(list(json.load(fp).keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}

        # 读入文件的文件名
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.tensorflow = "_tf" in file_list_name # tensorflow version of data/返回一个逻辑值
        self.normalization = normalization #逻辑值
        self.mesh_pos = mesh_pos  #？？？？？？
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border #逻辑值

    def __getitem__(self, index):
        if self.tensorflow:
            filename = self.file_names[index][17:]
            #意为如果是tf的话 去掉 file_names中开头的17个字母Data/ShapeNetP2M/
            label = filename.split("/", maxsplit=1)[0]
            #maxsplit=1 means only split it into tow parts and preserve the first part by using [0]
            #self.file_root = shapenet
            pkl_path = os.path.join(self.file_root, "data_tf", filename)
            img_path = pkl_path[:-4] + ".png"

            #得到pts和normals
            with open(pkl_path) as f:
                data = pickle.load(open(pkl_path, 'rb'), encoding="latin1")
            #.dat文件里到底是什么？？？？？？？
            #取所有行  前三列是pts points的坐标？ 后三列是normal 点的向量？
            pts, normals = data[:, :3], data[:, 3:]

            #得到img
            img = io.imread(img_path)
            #将alpha维度值为0的地方转变为255
            img[np.where(img[:, :, 3] == 0)] = 255
        
            #用什么方法resize图片
            if self.resize_with_constant_border:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                       mode='constant', anti_aliasing=False)  # to match behavior of old versions
            else:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

            #丢掉alpha维度？
            img = img[:, :, :3].astype(np.float32)

        else:
            label, filename = self.file_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]

        #为什么需要减去mesh_pos？ mesh_pos是什么？？？
        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "filename": filename,
            "length": length
        }

    def __len__(self):
        return len(self.file_names)



class AFLW2000(BaseDataset):
    """
    Dataset wrapping images and target meshes for AFLW2000.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
        super().__init__()
        self.file_root = file_root

        #INPUT ARGUMENTS
        #file_root: 
        #  original:..datasets/data/shapenet
        #  now:..datasets/data/AFLW2000-3D

        #file_list_name:
        #  original:e.g. train_all.txt
        #  now: train_aflw.txt/ test_aflw.txt

        #mesh_pos:
        #  original:[0., 0., -0.8]
        #  now:???do we need mesh_pos? how can we determine the mesh_pos?????????????

        #normalization:...

        #shapenet_options:...


        #SELF VARIABLES
        #self.labels_map: {'02691156': 0, '02828884': 1, '02933112': 2, '02958343': 3, '03001627': 4, '03211117': 5, '03636649': 6, '03691459': 7, '04090263': 8, '04256520': 9, '04379243': 10, '04401088': 11, '04530566': 12}
        self.labels_map = {'face':0}

        #self.file_names: ['02691156_fff513f407e00e85a9ced22d91ad7027_19.dat', '02691156_fff513f407e00e85a9ced22d91ad7027_20.dat', '02691156_fff513f407e00e85a9ced22d91ad7027_23.dat']
        #self.file_names: ['image02795.jpg']
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")

        self.normalization = normalization #boolean 
        self.mesh_pos = mesh_pos #mesh position


    def __getitem__(self, index):

        label = "face"
        filename = self.file_names[index] #file name of the img. e.g.image00002.jpg

        img_path =  self.file_root + "/AFLW2000/"+ filename
        data_path = self.file_root + "/AFLW2000/"+ filename[:-4] + ".txt"

        #np.loadtxt need file without "," !!!!!!!!
        data = np.loadtxt(data_path)
        #first 3 columns is point positions?
        pts, normals = data[:, :3], data[:, 3:]

        img = io.imread(img_path)

        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                    mode='constant', anti_aliasing=False)  # to match behavior of old versions
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

        img = img.astype(np.float32)

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1))) #turn data into [channels, height, width]
        img_normalized = self.normalize_img(img) if self.normalization else img  #normalize_img哪来的????????????????


        #OUTPUT QUESTIONS:
        #labels: 0????????????
        #filename:????????????

        return {
            "images": img_normalized,
            "images_orig": img,
            "points": pts,
            "normals": normals,
            "labels": 0, #means face
            "filename": filename,#dont know if its used later?????
            "length": length
        }

    def __len__(self):
        return len(self.file_names)


#ImageFolder啥意思
#也是一个dataset class
#返回img
class ShapeNetImageFolder(BaseDataset):

    def __init__(self, folder, normalization, shapenet_options):
        #super？
        super().__init__()
        self.normalization = normalization
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.file_list = []
        for fl in os.listdir(folder):
            file_path = os.path.join(folder, fl)
            # check image before hand
            try:
                if file_path.endswith(".gif"):
                    raise ValueError("gif's are results. Not acceptable")
                Image.open(file_path)
                self.file_list.append(file_path)
            except (IOError, ValueError):
                print("=> Ignoring %s because it's not a valid image" % file_path)

    def __getitem__(self, item):
        img_path = self.file_list[item]
        img = io.imread(img_path)

        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255

        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "filepath": self.file_list[item]
        }

    def __len__(self):
        return len(self.file_list)


def get_shapenet_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """
    def shapenet_collate(batch):
        #batch type: a list?
        if len(batch) > 1:
            all_equal = True
            for t in batch:
                if t["length"] != batch[0]["length"]:
                    all_equal = False
                    break
            points_orig, normals_orig = [], []
            if not all_equal:
                for t in batch:
                    pts, normal = t["points"], t["normals"]
                    length = pts.shape[0]
                    choices = np.resize(np.random.permutation(length), num_points)
                    t["points"], t["normals"] = pts[choices], normal[choices]
                    points_orig.append(torch.from_numpy(pts))
                    normals_orig.append(torch.from_numpy(normal))
                ret = default_collate(batch)
                ret["points_orig"] = points_orig
                ret["normals_orig"] = normals_orig
                return ret
        ret = default_collate(batch)
        ret["points_orig"] = ret["points"]
        ret["normals_orig"] = ret["normals"]
        return ret

    return shapenet_collate