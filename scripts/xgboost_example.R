# XGBoost draft
# Maybe shift aroound and start with caret
# Looks like data is for regression

library(caret)
library(dplyr)
library(ggplot2)
library(glue)
library(ModelMetrics)
library(OpenMPController) # for Kaggle backend
library(readr)
library(vtreat)
library(xgboost)

set.seed(57)
omp_set_num_threads(4) # caret parallel processing threads

tr <- read_csv("../input/train.csv", col_types = cols(SalePrice = "d"))
te <- read_csv("../input/test.csv")



treat_plan <- vtreat::designTreatmentsZ(
  dframe = tr, # training data
  varlist = colnames(tr) %>% .[. != "Id"], # input variables = all training data columns, except id
  codeRestriction = c("clean", "isBAD", "lev"), # derived variables types (drop cat_P)
  verbose = FALSE) # suppress messages


score_frame <- treat_plan$scoreFrame %>% 
  select(varName, origName, code)

head(score_frame)

unique(score_frame$code)

# list of variables without the target variable
te$SalePrice <- as.numeric(-1)

tr_treated <- vtreat::prepare(treat_plan, tr)
te_treated <- vtreat::prepare(treat_plan, te)

tr_treated$SalePrice_clean <- log(tr_treated$SalePrice_clean)

dim(tr_treated)

tr_holdout <- dplyr::sample_frac(tr_treated, 0.2)
hid <- as.numeric(rownames(tr_holdout))
tr_treated <- tr_treated[-hid, ]

ggplot2::qplot(tr_holdout$SalePrice_clean, main="Hold-out Set") + geom_histogram(colour="black", fill="grey") + theme_bw() 

ggplot2::qplot(tr_treated$SalePrice_clean, main="Training Set") + geom_histogram(colour="black", fill="grey") + theme_bw() 

input_x <- as.matrix(select(tr_treated, -SalePrice_clean))
input_y <- tr_treated$SalePrice_clean


#### 3 Using caret

# Go through 'hyper parameters'

# OLS as baseline model
ggplot2::qplot(tr$SalePrice) + geom_histogram(colour="black", fill="grey") + theme_bw()

ggplot2::qplot(tr_treated$SalePrice_clean) + geom_histogram(colour="black", fill="grey") + theme_bw()


(mcor <- tr_treated %>% 
    {cor(select(., -SalePrice_clean), .$SalePrice_clean, method = "spearman")} %>% 
    .[. == max(.), ])


lin_x <- tr_treated[, which(names(tr_treated) == names(mcor))]
lin_y <- tr_treated$SalePrice_clean

ggplot2::ggplot() +
  aes(x = lin_x, y = lin_y) +
  geom_jitter() +
  xlab(names(mcor)) +
  ylab("SalePrice (log)") +
  theme_bw() +
  geom_smooth(method = "lm")


linear_base <- lm(paste0("SalePrice_clean ~ ", names(mcor)),data = tr_treated)


grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

xgb_base <- caret::train(
  x = input_x,
  y = input_y,
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = TRUE
)


# Grid search hyperparameters

nrounds <- 1000

# note to start nrounds from 200, as smaller learning rates result in errors so
# big with lower starting points that they'll mess the scales
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

xgb_tune <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE
)


# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

tuneplot(xgb_tune)


xgb_tune$bestTune


# Max child weight etc...
tune_grid2 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = ifelse(xgb_tune$bestTune$max_depth == 2,
                     c(xgb_tune$bestTune$max_depth:4),
                     xgb_tune$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3),
  subsample = 1
)

xgb_tune2 <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid2,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune2)


xgb_tune2$bestTune

# Column and row sampling
tune_grid3 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_tune3 <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid3,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune3, probs = .95)


xgb_tune3$bestTune


# Gamma (explain)

tune_grid4 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune4 <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid4,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune4)
xgb_tune4$bestTune


# Reducing the learning rate

tune_grid5 <- expand.grid(
  nrounds = seq(from = 100, to = 10000, by = 100),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = xgb_tune4$bestTune$gamma,
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune5 <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  tuneGrid = tune_grid5,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune5)

xgb_tune5$bestTune

# Fitting the model

(final_grid <- expand.grid(
  nrounds = xgb_tune5$bestTune$nrounds,
  eta = xgb_tune5$bestTune$eta,
  max_depth = xgb_tune5$bestTune$max_depth,
  gamma = xgb_tune5$bestTune$gamma,
  colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
  min_child_weight = xgb_tune5$bestTune$min_child_weight,
  subsample = xgb_tune5$bestTune$subsample
))


(xgb_model <- caret::train(
  x = input_x,
  y = input_y,
  trControl = train_control,
  tuneGrid = final_grid,
  method = "xgbTree",
  verbose = TRUE
))


# Evaluating performance

holdout_x <- select(tr_holdout, -SalePrice_clean)
holdout_y <- tr_holdout$SalePrice_clean

(linear_base_rmse <- ModelMetrics::rmse(holdout_y, predict(linear_base, newdata = holdout_x)))

(xgb_base_rmse <- ModelMetrics::rmse(holdout_y, predict(xgb_base, newdata = holdout_x)))

(xgb_model_rmse <- ModelMetrics::rmse(holdout_y, predict(xgb_model, newdata = holdout_x)))
















