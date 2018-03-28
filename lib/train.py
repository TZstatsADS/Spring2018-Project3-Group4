from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
import pickle as pkl
import numpy as np
import time
from sklearn.model_selection import RepeatedKFold
import argparse
from functools import reduce

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




def main(data_dir, hidden_unit, epochs, batch_size, all_data):
    # load data, only the training set
    with open(data_dir, "rb") as input_file:
        X_train,y_train, X_val, y_val = pkl.load(input_file)
    X_train = X_train.astype(np.float32)/255
    X_val = X_val.astype(np.float32)/255
    
    # change to one hot
    y_train_OH = np.zeros((y_train.shape[0],3))
    y_train_OH[np.arange(y_train.shape[0]), y_train] = 1
    
    y_val_OH = np.zeros((y_val.shape[0],3))
    y_val_OH[np.arange(y_val.shape[0]), y_val] = 1
    
    # shuffle the data
    np.random.seed(10)
    num_train = X_train.shape[0]
    index = np.random.choice(num_train,num_train,replace=False)
    X_train = X_train[index]
    y_train_OH = y_train_OH[index]
    
    if all_data:
        X_watch_val = X_val
        y_watch_val = y_val_OH
        
        X_train_in = X_train
        y_train_in = y_train_OH
    else:
        num_watch_val = int(X_train.shape[0]*0.1)

        X_train_in = X_train[num_watch_val:]
        y_train_in = y_train_OH[num_watch_val:]

        X_watch_val = X_train[:num_watch_val]
        y_watch_val = y_train_OH[:num_watch_val]

    # load pretrained model
    model = define_model(hidden_unit=hidden_unit)
    datagen = ImageDataGenerator(
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True,
                                zoom_range = 0.5,
                                fill_mode = 'nearest'
                                )

    datagen.fit(X_train_in)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # save the model with best watch_val loss
    if all_data:
        train_time = str(int(time.time()))
        model_save_to = '../output/keras_model/mobilenet_'+train_time+'.hdf5'
    else:
        model_save_to = '../output/keras_model/mobilenet_partial_data.hdf5'
    checkpointer = ModelCheckpoint(filepath=model_save_to, verbose=1, save_best_only=True)

    # train the model on the new data for a few epochs
    print('train on', X_train_in.shape[0], 'data.')
    hist = model.fit_generator(datagen.flow(X_train_in, y_train_in, batch_size=batch_size), 
                               validation_data=(X_watch_val, y_watch_val),
                               steps_per_epoch=len(X_train_in) / batch_size, 
                               epochs=epochs,shuffle = True,callbacks=[checkpointer])

    model.load_weights(model_save_to)
    # val accuracy
    val_acc = np.mean(np.argmax(model.predict(X_val),axis=1)==np.argmax(y_val_OH, axis=1))
    print('validation accuracy:',val_acc)

            

            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cross Validation")
    parser.add_argument(
        "--data_dir", help="data of directory",
        default='../output/data/data224.pkl')
    parser.add_argument(
        "--hidden_unit", help="hidden unit size",
        default=1024,type=int)
    parser.add_argument(
        "--epochs", help="training epochs",
        default=20,type=int)
    parser.add_argument(
        "--batch_size", help="batch size",
        default=32,type=int)
    parser.add_argument(
        "--all_data", help="Training on all data provide",
        default=1,type=int)
    
    args = parser.parse_args()
    main(args.data_dir, args.hidden_unit, args.epochs, args.batch_size, args.all_data)
            
            
            
            