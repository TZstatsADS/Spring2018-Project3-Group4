#############################################
### Main execution script for experiments ###
#############################################

### Author: Group 4
### Project 3
### ADS Spring 2018



#####################
#  Advanced Model:  #
#    Fine Tuning    #
#     MobileNet     #
#####################

### Resize image and seperate train and validation according to label2.csv
tm_feature_train <- system.time(
  system('python ../lib/preprocessing.py --img_size=224 --train 1 --lab_dir ../output/label2.csv --img_dir ../data/train/images --out_dir ../output/data'))

### Cross Validation on the number of hidden unit in the last Dense layer
system('python ../lib/cross_validation.py --k 5 --hidden_unit_list 256 512 1024 --batch_size 128')

### Training on all provided data and save the model
tm_train <- system.time(
  system('python ../lib/train.py --hidden_unit 256 --epochs 30 --all_data 1'))

### Resize image from test set
tm_feature_test <- system.time(
  system('python ../lib/preprocessing.py --img_size=224 --train 0 --img_dir ../data/test/images --lab_dir ../data/test/label.csv'))

### Predict on new data use trained model and save the result to label.csv
tm_test <- system.time(
  system('python ../lib/test.py --hidden_unit 256 --model_path ../output/keras_model/mobilenet_1522197600.hdf5 '))

### Summarize Running Time
cat('Fine tuning on MobileNet (imagenet pretraine)')
cat("Time for constructing training features=", tm_feature_train[1], "s \n")
cat("Time for constructing testing features=", tm_feature_test[1], "s \n")
cat("Time for training model=", tm_train[1], "s \n")
cat("Time for making prediction=", tm_test[1], "s \n")

