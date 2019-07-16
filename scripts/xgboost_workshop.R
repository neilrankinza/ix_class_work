# XGBoost workshop
# Neil Rankin
# 2018/07/16


# Load packages
# https://cran.r-project.org/web/packages/vtreat/vignettes/vtreat.html
# prepares real-world data for predictive modeling in a statistically sound manner
library(vtreat)
library(caret)
library(tidyverse)

# Set seed
set.seed(1234)

# House price data for Kaggle competition
# Read in data

tr <- read_csv("data/train.csv", col_types = cols(SalePrice = "d"))
te <- read_csv("data/test.csv")

# Create a 'plan' which transforms the data and can be followed for test data
# and 'production' data

treat_plan <- vtreat::designTreatmentsZ(
  dframe = tr, # training data
  varlist = colnames(tr) %>% .[. != "Id"], # input variables = all training data columns, except id
  codeRestriction = c("clean", "isBAD", "lev"), # derived variables types (drop cat_P)
  verbose = FALSE) # suppress messages


score_frame <- treat_plan$scoreFrame %>% 
  select(varName, origName, code)

head(score_frame)

unique(score_frame$code)

# clean stands for cleaned numerical variable, isBAD indicates that a value
# replacement has occurred (which indicates a missing value in this case), and
# lev is a binary indicator whether a particular value of that categorical
# variable was present.

# list of variables without the target variable
te$SalePrice <- as.numeric(-1)

tr_treated <- vtreat::prepare(treat_plan, tr)
te_treated <- vtreat::prepare(treat_plan, te)

# Why do we log?
tr_treated <- tr_treated %>% 
  mutate(SalePrice_clean = log(SalePrice)) %>% 
  select(-SalePrice)
         
         
# Create a hold-out dataset (to test our model before we 'contaminate' the test
# data)

tr_holdout <- dplyr::sample_frac(tr_treated, 0.2)
hid <- as.numeric(rownames(tr_holdout))
tr_treated <- tr_treated[-hid, ]

# A look at whether outcome variables look the same

ggplot2::qplot(tr_holdout$SalePrice_clean, main="Hold-out Set") + 
  geom_histogram(colour="black", fill="grey") + theme_bw() 

ggplot2::qplot(tr_treated$SalePrice_clean, main="Training Set") + 
  geom_histogram(colour="black", fill="grey") + theme_bw() 



# A simple linear model
model_lm1 <- train(SalePrice_clean ~ ., data=tr_treated, method='lm')
predicted_lm1 <- predict(model_lm1, tr_holdout)

# Use caret::postResample function for model performance
postResample(pred = predicted_lm1, obs = tr_holdout$SalePrice_clean)
varImp(model_lm1)

# XGBoost 1
# use nthread to parallelise
model_xgb1 <- train(SalePrice_clean ~ ., data=tr_treated, nthread = 2, method='xgbLinear')
predicted_xgb1 <- predict(model_xgb1, tr_holdout)

# Some model performance parameters
postResample(pred = predicted_xgb1, obs = tr_holdout$SalePrice_clean)
varImp(model_xgb1)

library(MLmetrics)
RMSLE(y_pred = exp(predicted_xgb1), y_true = exp(tr_holdout$SalePrice_clean))

# XGBoost 2 - tune some parameters
# Do this with a tree since there is more scope for parameter tuning


tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)


xgb_tune <- caret::train(SalePrice_clean ~ ., data=tr_treated, 
                         method='xgbTree', verbose = TRUE, 
                         trControl = tune_control, 
                         tuneGrid = tune_grid, 
                         nthread = 4)


# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

tuneplot(xgb_tune)

xgb_tune$bestTune

# gamma minimum reduction in loss function required for node to split


tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = c(2),
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)


xgb_tune <- caret::train(SalePrice_clean ~ ., data=tr_treated, 
                         method='xgbTree', verbose = TRUE, 
                         trControl = tune_control, 
                         tuneGrid = tune_grid, 
                         nthread = 4)


# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

tuneplot(xgb_tune)

xgb_tune$bestTune

# Try and tune some of the other parameters and see whether you can improve the model

# Once you have gotten your 'best' model run it on the test data



# Tuning on linear - not much goes on here
# Code left in for illustrative purposes

nrounds <- 1000
# note to start nrounds from 200, as smaller learning rates result in errors so
# big with lower starting points that they'll mess the scales
tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50), 
  lambda = 0, 
  alpha = 0, 
  eta = c(0.025, 0.05, 0.1, 0.3)
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

xgb_tune <- caret::train(SalePrice_clean ~ ., data=tr_treated, 
                         method='xgbLinear', verbose = TRUE, 
                         trControl = tune_control, 
                         tuneGrid = tune_grid, nthread = 2)


# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

tuneplot(xgb_tune)

xgb_tune$bestTune
