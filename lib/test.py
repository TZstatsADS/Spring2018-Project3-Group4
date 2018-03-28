from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
import pickle as pkl
import numpy as np
import argparse
import pandas as pd

def define_model(hidden_unit=1024):
    # create the base pre-trained model
    base_model = MobileNet(input_shape=(224, 224, 3),weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(hidden_unit, activation='relu')(x)
    # and a logistic layer 
    predictions = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    return model

def main(hidden_unit, model_path, data_dir, lab_dir, out_dir):
    # load data
    with open(data_dir, "rb") as input_file:
        X_test = pkl.load(input_file)
    X_test = X_test.astype(np.float32)/255
    
    # load label.csv
    label = pd.read_csv(lab_dir,index_col=0)
    
    model = define_model(hidden_unit)
    model.load_weights(model_path)
    
    result = np.argmax(model.predict(X_test),axis=1)+1
    label['label'] = result
    label.to_csv(out_dir)
    print('label.csv save to',out_dir)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cross Validation")
    parser.add_argument(
        "--model_path", help="training epochs",
        required=True)
    parser.add_argument(
        "--data_dir", help="data of directory",
        default='../output/data/test_data224.pkl')
    parser.add_argument(
        "--hidden_unit", help="hidden unit size",
        default=256,type=int)
    parser.add_argument(
        "--lab_dir", help="Path to label.csv file",
        default='../data/test/label.csv')
    parser.add_argument(
        "--out_dir", help="output path of label.csv",
        default='../output/label.csv')
    
    args = parser.parse_args()
    main(args.hidden_unit, args.model_path, args.data_dir, args.lab_dir, args.out_dir)
    
    
    