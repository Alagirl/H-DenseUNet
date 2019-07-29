# -*- coding: utf-8 -*-
from medpy.io import load, save
import os
import os.path
import numpy as np
from tqdm import tqdm

datadir = '/home/zhounan/Documents/dataset/ISBI2017(nii)/'

def proprecessing(image_path, save_folder):
    # 生成处理后的nii文件
    if not os.path.exists("data/"+save_folder):
        os.mkdir('data/' + save_folder)
    filelist = os.listdir(image_path)
    filelist = [item for item in filelist if 'volume' in item]
    for file in tqdm(filelist):
        img, img_header = load(image_path+file)
        img[img < -200] = -200
        img[img > 250] = 250
    img = np.array(img, dtype='float32')
    print("Saving image "+file)
    save(img, './data/' + save_folder + file)

def generate_livertxt(image_path, save_folder):
    # 每个case一个存有liver位置的txt文件
    if not os.path.exists("data/"+save_folder):
        os.mkdir("data/"+save_folder)

    # Generate Livertxt
    if not os.path.exists("data/"+save_folder+'LiverPixels'):
        os.mkdir("data/"+save_folder+'LiverPixels')

    for i in xrange(0, 131):
        livertumor, header = load(image_path+'segmentation-'+str(i)+'.nii')
        f = open('data/' + save_folder + '/LiverPixels/liver_' + str(i) + '.txt', 'w')
        index = np.where(livertumor == 1)
        x = index[0]
        y = index[1]
        z = index[2]
        np.savetxt(f, np.c_[x, y, z], fmt="%d")
        f.write("\n")
        f.close()

def generate_tumortxt(image_path, save_folder):
    # 每个case一个存有tumor位置的txt文件
    if not os.path.exists("data/"+save_folder):
        os.mkdir("data/"+save_folder)

    # Generate Livertxt
    if not os.path.exists("data/"+save_folder+'TumorPixels'):
        os.mkdir("data/"+save_folder+'TumorPixels')

    for i in xrange(0, 131):
        livertumor, header = load(image_path+'segmentation-'+str(i)+'.nii')
        f = open("data/"+save_folder+"/TumorPixels/tumor_"+str(i)+'.txt', 'w')
        index = np.where(livertumor == 2)

        x = index[0]
        y = index[1]
        z = index[2]

        np.savetxt(f, np.c_[x, y, z], fmt="%d")

        f.write("\n")
        f.close()

def generate_txt(image_path, save_folder):
    # 每个case一个存有liver bounding box在x,y,z三维上的最值的txt文件
    if not os.path.exists("data/"+save_folder):
        os.mkdir("data/"+save_folder)

    # Generate Livertxt
    if not os.path.exists("data/"+save_folder+'LiverBox'):
        os.mkdir("data/"+save_folder+'LiverBox')
    for i in xrange(0, 131):
        values = np.loadtxt("data/" + 'myTrainingDataTxt/LiverPixels/liver_' + str(i) + '.txt', delimiter=' ', usecols=[0, 1, 2])
        a = np.min(values, axis=0)
        b = np.max(values, axis=0)
        box = np.append(a, b, axis=0)
        np.savetxt('data/myTrainingDataTxt/LiverBox/box_'+str(i)+'.txt', box, fmt='%d')


train_path='/home/zhounan/Documents/dataset/ISBI2017(nii)/Training_Batch_2/'
test_path='/home/zhounan/Documents/dataset/ISBI2017(nii)/Training_Batch_1/'
proprecessing(image_path=train_path, save_folder='myTrainingData/')
proprecessing(image_path=test_path, save_folder='myTestData/')
print("Generate liver txt ")
generate_livertxt(image_path=train_path, save_folder='myTrainingDataTxt/')
print("Generate tumor txt")
generate_tumortxt(image_path=train_path, save_folder='myTrainingDataTxt/')
print("Generate liver box ")
generate_txt(image_path=train_path, save_folder='myTrainingDataTxt/')
