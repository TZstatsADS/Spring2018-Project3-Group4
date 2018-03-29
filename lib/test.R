######################################################
### Fit the classification model with testing data ###
######################################################

### Author: Group 4
### Project 3
### ADS Spring 2018

test <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  
  pred <- predict(fit_train$fit, newdata=dat_test, 
                  n.trees=fit_train$iter, type="response")
  
  return(as.numeric(pred> 0.5))
}

xgb_pred <- function(model,test){
  test$label <- test$label-1
  test_smm <- sparse.model.matrix(label~.,data = test)
  pred <- predict(model,test_smm)
  err <- mean(pred!=test$label)
  return(list("pred"=pred+1, "err" = err))
}

