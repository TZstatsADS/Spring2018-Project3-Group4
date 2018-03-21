xgb_model <- function(train,par){
  param <- list(
    # General Parameters
    booster            = "gbtree",          # default
    silent             = 0,                 # default
    # Booster Parameters
    eta                = par$eta,           # default = 0.30
    gamma              = 0,                 # default
    max_depth          = par$max_depth,     # default = 6
    min_child_weight   = 1,                 # default
    subsample          = 1,                 # default = 1
    colsample_bytree   = 1,                 # default = 1
    num_parallel_tree  = 1,                 # default
    lambda             = 0,                 # default
    lambda_bias        = 0,                 # default
    alpha              = 0,                 # default
    # Task Parameters
    objective          = "multi:softmax",   # default = "reg:linear"
    num_class          = 3,                 # default = 0
    base_score         = 0.5  ,               # default
    eval_metric        = "merror"           # default = "rmes"
  )
  
  # convert train dataframe into a design matrix
  train_smm <- sparse.model.matrix(label ~ ., data = train)
  train.xgb <- xgb.DMatrix(data = train_smm, label = train$label-1)
  
  set.seed(3)
  
  # train xgb model
  model <- xgb.train(params = param,
                     data = train.xgb,
                     nrounds = 100,
                     verbose = F,
                     watchlist = list(train_smm=train.xgb))
  return(model)
}



xgb_param <- function(train,K){
  
  # convert train dataframe into a design matrix
  train_smm <- sparse.model.matrix(label ~ ., data = train)
  train.xgb <- xgb.DMatrix(data = train_smm, label = train$label-1)
  
  eta <- seq(0.05,0.3,0.05)
  max_depth <- seq(3,8)
  best_param <- list()
  best_err <- Inf
  param_mat <- matrix(nrow = length(eta),ncol = length(max_depth))
  
  for(i in 1:nrow(param_mat)){
    for(j in 1:ncol(param_mat)){
      my.param <- list(eta = eta[1], max_depth = max_depth[j])
      set.seed(3)
      cv.result <- xgb.cv(data = train.xgb, 
                          params = my.param,
                          nrounds = 100,
                          gamma = 0, 
                          subsample = 1,
                          
                          # Task Parameters
                          objective = "multi:softmax",   # default = "reg:linear"
                          num_class = 3,
                          
                          nfold = K,
                          nthread = 2,
                          verbose = F,
                          maximize = F,
                          prediction = T)
      min_err <- min(cv.result$evaluation_log$test_merror_mean)
      param_mat[i,j] <- min_err
      
      if(min_err < best_err){
        best_param <- my.param
        best_err <- min_err
      }
      
    }
  }
  
  return(list("error" = param_mat,
              "best_param" = best_param,
              "best_err"= best_err))
}


xgb_pred <- function(model,test){
  test$label <- test$label-1
  test_smm <- sparse.model.matrix(label~.,data = test)
  pred <- predict(model,test_smm)
  err <- mean(pred!=test$label)
  return(list("pred"=pred+1, "err" = err))
}
