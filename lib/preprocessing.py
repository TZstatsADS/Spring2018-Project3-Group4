import pickle as pkl
import pandas as pd
import numpy as np
import os
import cv2
from skimage import transform
import argparse

def main(img_size=224,
         lab_dir='../output/label2.csv',
         img_dir='../data/train/images',
         out_dir='../output/data/',
         train = True):
    if train:
        print('Start preprocessing training data')
        label = pd.read_csv(lab_dir,index_col=0)
        y_train = np.array(label[label.train==1].label.tolist())
        y_val = np.array(label[label.train==0].label.tolist())
        y_train -= 1
        y_val -= 1
        X_train = []
        X_val = []
        for i,r in label.iterrows():
            #print (r.iloc[1])
            img = cv2.imread(img_dir+'/'+r.iloc[0][-4:]+'.jpg')
            img = transform.resize(img,[img_size,img_size]).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if r.iloc[2] == 1:
                X_train.append(img)
            else:
                X_val.append(img)
        X_train = np.array(X_train,dtype=np.float32)
        X_val = np.array(X_val,dtype=np.float32)
        X_train = 255 * X_train
        X_val = 255 * X_val
        X_train = X_train.astype(np.uint8)
        X_val = X_val.astype(np.uint8)
        with open(out_dir+'/data'+str(img_size)+'.pkl', "wb") as output_file:
             pkl.dump((X_train, y_train, X_val, y_val), output_file)

        print('shape of (X_train, y_train, X_val, y_val):',X_train.shape, y_train.shape, X_val.shape, y_val.shape)
        print('finished')
        
    else:
        X_test = []
        for file in os.dirlist(img_dir):
            if file[-4:] is not '.jpg':
                continue
            else:
                img = cv2.imread(os.path.join(img_dir,file))
                img = transform.resize(img,[224,224]).astype(np.float32)
                img = transform.resize(img,[img_size,img_size]).astype(np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X_test.append(img)
        X_test = np.arrag(X_test,dtype=np.float32)
        X_test = 255 * X_test
        X_test = X_test.astype(np.uint8)
        with open(out_dir+'/test_data'+str(img_size)+'.pkl', "wb") as output_file:
             pkl.dump((X_train, y_train, X_val, y_val), output_file)
        print('shape of X_test:', X_test.shape)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument(
        "--img_size", help="resize the images",
        default=224, type=int, required=True)
    parser.add_argument(
        "--lab_dir", help="Path to label.csv file",
        default='../output/label2.csv')
    parser.add_argument(
        "--img_dir", help="Path to the images directory",
        default='../data/train/images')
    parser.add_argument(
        "--out_dir", help="Path to output directory",
        default='../output/data')
    parser.add_argument(
        "--train", help="Is it the train data or the test data",
        default=True,type=bool)
    
    args = parser.parse_args()
    main(img_size=args.img_size,
         lab_dir=args.lab_dir,
         img_dir=args.img_dir,
         out_dir=args.out_dir,
         train=args.train)