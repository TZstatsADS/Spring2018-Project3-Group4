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




def main(data_dir, k, hidden_unit_list, epochs, batch_size):
    # load data, only the training set
    with open(data_dir, "rb") as input_file:
        X_train,y_train, _, _ = pkl.load(input_file)
    X_train = X_train.astype(np.float32)/255
    
    # change to one hot
    y_train_OH = np.zeros((y_train.shape[0],3))
    y_train_OH[np.arange(y_train.shape[0]), y_train] = 1
    
    # shuffle the data
    np.random.seed(10)
    num_train = X_train.shape[0]
    index = np.random.choice(num_train,num_train,replace=False)
    X_train = X_train[index]
    y_train_OH = y_train_OH[index]
    
    cv_scores = []
    cv_scores_mean = []

    for hu in hidden_unit_list:
        print('testing hidden unit =', hu)
        # cv
        rkf = RepeatedKFold(n_splits=k, n_repeats=1, random_state=2652124)
        scores = []
        cv = 0
        for train_index, test_index in rkf.split(X_train):
            cv += 1
            print('CV:', str(cv)+'/'+str(k))
            X_val_CV = X_train[test_index]
            y_val_CV = y_train_OH[test_index]
            
            # seperate watch validation set for finding the model has better generization
            num_watch_val = int(train_index.shape[0]*0.1)
            
            X_train_CV = X_train[train_index[num_watch_val:]]
            y_train_CV = y_train_OH[train_index[num_watch_val:]]
            
            X_watch_val = X_train[train_index[:num_watch_val]]
            y_watch_val = y_train_OH[train_index[:num_watch_val]]
            
            # load pretrained model
            model = define_model(hidden_unit=hu)
            datagen = ImageDataGenerator(
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        zoom_range = 0.5,
                                        fill_mode = 'nearest'
                                        )
            
            datagen.fit(X_train_CV)
            
            # compile the model (should be done *after* setting layers to non-trainable)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
            
            # save the model with best watch_val loss
            checkpointer = ModelCheckpoint(filepath='../output/keras_model/cv_tmp.hdf5', verbose=1, save_best_only=True)
            
            # train the model on the new data for a few epochs
            hist = model.fit_generator(datagen.flow(X_train_CV, y_train_CV, batch_size=batch_size), 
                                       validation_data=(X_watch_val, y_watch_val),
                                       steps_per_epoch=len(X_train_CV) / batch_size, 
                                       epochs=epochs,shuffle = True,callbacks=[checkpointer])
            
            model.load_weights('../output/keras_model/cv_tmp.hdf5')
            # val accuracy
            val_CV_acc = np.mean(np.argmax(model.predict(X_val_CV),axis=1)==np.argmax(y_val_CV, axis=1))
            scores.append(val_CV_acc)
            
        cv_scores.append(scores)
        cv_scores_mean.append(reduce(lambda x, y: x + y, scores) / len(scores))
        
    print(cv_scores)
    print(cv_scores_mean)
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cross Validation")
    parser.add_argument(
        "--data_dir", help="data of directory",
        default='../output/data/data224.pkl')
    parser.add_argument(
        "--k", help="k fold",
        default=5,type=int)
    parser.add_argument(
        "--hidden_unit_list", help="hidden unit size",
        default=[512,1024],nargs='+',type=int)
    parser.add_argument(
        "--epochs", help="training epochs",
        default=20,type=int)
    parser.add_argument(
        "--batch_size", help="batch size",
        default=32,type=int)
    
    args = parser.parse_args()
    main(args.data_dir, args.k, args.hidden_unit_list, args.epochs, args.batch_size)
            
            
            
            