

#https://www.kaggle.com/nschneider/gbm-vs-xgboost-vs-lightgbm

# Libraries
library(pROC, quietly=TRUE)
library(microbenchmark, quietly=TRUE)

# Set seed so the train/test split is reproducible
set.seed(42)

# Read in the data and split it into train/test subsets
credit.card.data = read.csv("/home/terrence/CODING/Python/MODELS/creditcard.csv")

train.test.split <- sample(2
                           , nrow(credit.card.data)
                           , replace = TRUE
                           , prob = c(0.7, 0.3))
train = credit.card.data[train.test.split == 1,]
test = credit.card.data[train.test.split == 2,]

#+++++++++++++++++++ GBM ++++++++++++++++++++++++++++++

library(gbm, quietly=TRUE)

# Get the time to train the GBM model
system.time(
  gbm.model <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train, test)
                   , n.trees = 10
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train) / (nrow(train) + nrow(test))
  )
)
# Determine best iteration based on test data
best.iter = gbm.perf(gbm.model, method = "test")

# Get feature importance
gbm.feature.imp = summary(gbm.model, n.trees = best.iter)

# Plot and calculate AUC on test data
gbm.test = predict(gbm.model, newdata = test, n.trees = best.iter)
auc.gbm = roc(test$Class, gbm.test, plot = TRUE, col = "red")
print(auc.gbm)


#+++++++++++++++ XGBOOST +++++++++++++++++++++++++++++++++++++++++++


library(xgboost, quietly=TRUE)
xgb.data.train <- xgb.DMatrix(as.matrix(train[, colnames(train) != "Class"]), label = train$Class)
xgb.data.test <- xgb.DMatrix(as.matrix(test[, colnames(test) != "Class"]), label = test$Class)

# Get the time to train the xgboost model
xgb.bench.speed = microbenchmark(
  xgb.model.speed <- xgb.train(data = xgb.data.train
                               , params = list(objective = "binary:logistic"
                                               , eta = 0.1
                                               , max.depth = 3
                                               , min_child_weight = 100
                                               , subsample = 1
                                               , colsample_bytree = 1
                                               , nthread = 3
                                               , eval_metric = "auc"
                               )
                               , watchlist = list(test = xgb.data.test)
                               , nrounds = 10
                               , early_stopping_rounds = 40
                               , print_every_n = 20
  )
  , times = 5L
)
print(xgb.bench.speed)
print(xgb.model.speed$bestScore)


# Make predictions on test set for ROC curve
xgb.test.speed = predict(xgb.model.speed
                         , newdata = as.matrix(test[, colnames(test) != "Class"])
                         , ntreelimit = xgb.model.speed$bestInd)
auc.xgb.speed = roc(test$Class, xgb.test.speed, plot = TRUE, col = "blue")
print(auc.xgb.speed)



# Train a deeper xgboost model to compare accuarcy.
xgb.bench.acc = microbenchmark(
  xgb.model.acc <- xgb.train(data = xgb.data.train
                             , params = list(objective = "binary:logistic"
                                             , eta = 0.1
                                             , max.depth = 7
                                             , min_child_weight = 100
                                             , subsample = 1
                                             , colsample_bytree = 1
                                             , nthread = 3
                                             , eval_metric = "auc"
                             )
                             , watchlist = list(test = xgb.data.test)
                             , nrounds = 500
                             , early_stopping_rounds = 40
                             , print_every_n = 20
  )
  , times = 5L
)
print(xgb.bench.acc)
print(xgb.model.acc$bestScore)

#Get feature importance
xgb.feature.imp = xgb.importance(model = xgb.model.acc)


# Make predictions on test set for ROC curve
xgb.test.acc = predict(xgb.model.acc
                       , newdata = as.matrix(test[, colnames(test) != "Class"])
                       , ntreelimit = xgb.model.acc$bestInd)
auc.xgb.acc = roc(test$Class, xgb.test.acc, plot = TRUE, col = "blue")
print(auc.xgb.acc)


# xgBoost with Histogram
xgb.bench.hist = microbenchmark(
  xgb.model.hist <- xgb.train(data = xgb.data.train
                              , params = list(objective = "binary:logistic"
                                              , eta = 0.1
                                              , max.depth = 7
                                              , min_child_weight = 100
                                              , subsample = 1
                                              , colsample_bytree = 1
                                              , nthread = 3
                                              , eval_metric = "auc"
                                              , tree_method = "hist"
                                              , grow_policy = "lossguide"
                              )
                              , watchlist = list(test = xgb.data.test)
                              , nrounds = 500
                              , early_stopping_rounds = 40
                              , print_every_n = 20
  )
  , times = 5L
)
print(xgb.bench.hist)
print(xgb.model.hist$bestScore)


#Get feature importance
xgb.feature.imp = xgb.importance(model = xgb.model.hist)

# Make predictions on test set for ROC curve
xgb.test.hist = predict(xgb.model.hist
                        , newdata = as.matrix(test[, colnames(test) != "Class"])
                        , ntreelimit = xgb.model.hist$bestInd)
auc.xgb.hist = roc(test$Class, xgb.test.hist, plot = TRUE, col = "blue")
print(auc.xgb.hist)


#+++++++++++++++++++++ lightGBM ++++++++++++++++++++++++++++++++++++++

library(lightgbm, quietly=TRUE)
lgb.train = lgb.Dataset(as.matrix(train[, colnames(train) != "Class"]), label = train$Class)
lgb.test = lgb.Dataset(as.matrix(test[, colnames(test) != "Class"]), label = test$Class)

params.lgb = list(
  objective = "binary"
  , metric = "auc"
  , min_data_in_leaf = 1
  , min_sum_hessian_in_leaf = 100
  , feature_fraction = 1
  , bagging_fraction = 1
  , bagging_freq = 0
)

# Get the time to train the lightGBM model
lgb.bench = microbenchmark(
  lgb.model <- lgb.train(
    params = params.lgb
    , data = lgb.train
    , valids = list(test = lgb.test)
    , learning_rate = 0.1
    , num_leaves = 7
    , num_threads = 2
    , nrounds = 500
    , early_stopping_rounds = 40
    , eval_freq = 20
  )
  , times = 5L
)
print(lgb.bench)
print(max(unlist(lgb.model$record_evals[["test"]][["auc"]][["eval"]])))

# get feature importance
lgb.feature.imp = lgb.importance(lgb.model, percentage = TRUE)

# make test predictions
lgb.test = predict(lgb.model, data = as.matrix(test[, colnames(test) != "Class"]), n = lgb.model$best_iter)
auc.lgb = roc(test$Class, lgb.test, plot = TRUE, col = "green")
print(auc.lgb)


#+++++++++++++++++++++ SPEEED ++++++++++++++++++++++++++++++++++++++

print("GBM = ~243s")
print(xgb.bench.speed)
print(xgb.bench.hist)
print(lgb.bench)












