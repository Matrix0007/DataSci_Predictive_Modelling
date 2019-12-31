# knitr set
knitr::opts_chunk$set(eval = FALSE)

# Project Overview Visual
knitr::include_graphics("https://github.com/Matrix0007/datasets/blob/master/Project_Overview.png?raw=true")

#load packages if required
if(!require(mlbench)) install.packages("mlbench", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(Cubist)) install.packages("Cubist", repos = "http://cran.us.r-project.org")
if(!require(jpeg)) install.packages("jpeg", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

library(mlbench)
library(caret)
library(corrplot)
library(tibble)
library(Cubist)
library(jpeg)
library(readr)
library(tidyverse)
library(knitr)

#load data from url
dta <- url("http://course1.winona.edu/bdeppa/Stat%20425/Data/Boston_Housing.csv")
TBdta <- as.data.frame(read.csv(dta, check.names = FALSE))

# Split out validation dataset
# create a list of 80% of the rows in the original dataset we can use for training
set.seed(7)
validateIndex <- caret::createDataPartition(TBdta$MEDV, p=0.80, list=FALSE)
# select 20% of the data for validation
validateSet <- TBdta[-validateIndex,]
# use the remaining 80% of data to training and testing the models
DTset <- TBdta[validateIndex,]

# check dimensions of DTset
dim(DTset)

# review attribute types
sapply(DTset, class)

# data first 10 rows
head(DTset, n=10)

# summary of the attributes
summary(DTset)

# Correlation of the attributes
cor(DTset[,1:13])

# Boxplot visual of the attributes
par(mfrow=c(2,7))
for(i in 1:13){
  boxplot(DTset[,i], main=names(DTset)[i], col="deepskyblue", fg="mediumorchid", col.axis="darkgreen")
}

# histograms each attribute
par(mfrow=c(2,7))
for(i in 1:13) {
  hist(DTset[,i], main=names(DTset)[i], col="deepskyblue", fg="mediumorchid", col.axis="darkgreen")
}

# density plot for each attribute
par(mfrow=c(2,7))
for(i in 1:13) {
  plot(density(DTset[,i]), main=names(DTset)[i], col="blue", fg="mediumorchid", col.axis="darkgreen")
}

# Scatter Plots
pairs(DTset[,1:13], col="blue", fg="mediumorchid", col.axis="darkgreen")

# correlated plot
correlated<- cor(DTset[,1:13])
corrplot(correlated, type = "upper", order = "hclust", col = c("purple", "blue"), bg = "darkgrey")

# 10-fold cross validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

# LM
set.seed(7)
data.lm <- train(MEDV~., data=DTset, method="lm", metric=metric, preProc=c("center",
                                                                           "scale"), trControl=trainControl)
# GLM
set.seed(7)
data.glm <- train(MEDV~., data=DTset, method="glm", metric=metric, preProc=c("center",
                                                                             "scale"), trControl=trainControl)
# GLMNET
set.seed(7)
data.glmnet <- train(MEDV~., data=DTset, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=trainControl)
# SVM
set.seed(7)
data.svm <- train(MEDV~., data=DTset, method="svmRadial", metric=metric,
                  preProc=c("center", "scale"), trControl=trainControl)
# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
data.cart <- train(MEDV~., data=DTset, method="rpart", metric=metric, tuneGrid=grid,
                   preProc=c("center", "scale"), trControl=trainControl)
# KNN
set.seed(7)
data.knn <- train(MEDV~., data=DTset, method="knn", metric=metric, preProc=c("center",
                                                                             "scale"), trControl=trainControl)

# Review our algorithms
baseline.results <- resamples(list(LM=data.lm, GLM=data.glm, GLMNET=data.glmnet, SVM=data.svm,
                                   CART=data.cart, KNN=data.knn))
summary(baseline.results)

# Visualise via dotplot
dotplot(baseline.results)

# Exclude correlated attributes

# Look for correlated attributes
set.seed(7)
cutoff <- 0.70
correlations <- cor(DTset[,1:13])
CorrelatedData <- findCorrelation(correlations, cutoff=cutoff)
for(value in CorrelatedData)
{
  print(names(DTset)[value])
}
# create data exluding correlated features
DTsetFeatures <- DTset[,-CorrelatedData]
dim(DTsetFeatures)

# Algorithms with 10 fold cross validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(7)
data.lm <- train(MEDV~., data=DTsetFeatures, method="lm", metric=metric,
                 preProc=c("center", "scale"), trControl=trainControl)
# GLM
set.seed(7)
data.glm <- train(MEDV~., data=DTsetFeatures, method="glm", metric=metric,
                  preProc=c("center", "scale"), trControl=trainControl)
# GLMNET
set.seed(7)
data.glmnet <- train(MEDV~., data=DTsetFeatures, method="glmnet", metric=metric,
                     preProc=c("center", "scale"), trControl=trainControl)
# SVM
set.seed(7)
data.svm <- train(MEDV~., data=DTsetFeatures, method="svmRadial", metric=metric,
                  preProc=c("center", "scale"), trControl=trainControl)
# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
data.cart <- train(MEDV~., data=DTsetFeatures, method="rpart", metric=metric,
                   tuneGrid=grid, preProc=c("center", "scale"), trControl=trainControl)
# KNN
set.seed(7)
data.knn <- train(MEDV~., data=DTsetFeatures, method="knn", metric=metric,
                  preProc=c("center", "scale"), trControl=trainControl)
# Compare algorithms
feature.results <- resamples(list(LM=data.lm, GLM=data.glm, GLMNET=data.glmnet, SVM=data.svm,
                                  CART=data.cart, KNN=data.knn))
summary(feature.results)

# Visualise via dotplot
dotplot(feature.results)

# Run algorithms using 10-fold cross-validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(7)
data.lm <- train(MEDV~., data=DTset, method="lm", metric=metric, preProc=c("center",
                                                                           "scale", "BoxCox"), trControl=trainControl)
# GLM
set.seed(7)
data.glm <- train(MEDV~., data=DTset, method="glm", metric=metric, preProc=c("center",
                                                                             "scale", "BoxCox"), trControl=trainControl)
# GLMNET
set.seed(7)
data.glmnet <- train(MEDV~., data=DTset, method="glmnet", metric=metric,
                     preProc=c("center", "scale", "BoxCox"), trControl=trainControl)
# SVM
set.seed(7)
data.svm <- train(MEDV~., data=DTset, method="svmRadial", metric=metric,
                  preProc=c("center", "scale", "BoxCox"), trControl=trainControl)
# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
data.cart <- train(MEDV~., data=DTset, method="rpart", metric=metric, tuneGrid=grid,
                   preProc=c("center", "scale", "BoxCox"), trControl=trainControl)
# KNN
set.seed(7)
data.knn <- train(MEDV~., data=DTset, method="knn", metric=metric, preProc=c("center",
                                                                             "scale", "BoxCox"), trControl=trainControl)
# Compare algorithms
box.cox.results <- resamples(list(LM=data.lm, GLM=data.glm, GLMNET=data.glmnet, SVM=data.svm,
                                  CART=data.cart, KNN=data.knn))
summary(box.cox.results)

# Visualise via dotplot
dotplot(box.cox.results)

# Print DataSVM fiited model
print(data.svm)

# Tunning SVM and 
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
set.seed(7)
grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
data.svm <- train(MEDV~., data=DTset, method="svmRadial", metric=metric, tuneGrid=grid,
                  preProc=c("BoxCox"), trControl=trainControl)
print(data.svm)


# Ensembles
seed <- 7
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# RF
set.seed(seed)
ensemble.rf <- train(MEDV~., data=DTset, method="rf", metric=metric, preProc=c("BoxCox"),
                     trControl=trainControl)
# CUBIST
set.seed(seed)
ensemble.cubist <- train(MEDV~., data=DTset, method="cubist", metric=metric,
                         preProc=c("BoxCox"), trControl=trainControl)
# GBM
set.seed(seed)
ensemble.gbm <- train(MEDV~., data=DTset, method="gbm", metric=metric, preProc=c("BoxCox"),
                      trControl=trainControl, verbose=FALSE)

# Evaluate resultsalgorithms
ensembles.output <- resamples(list(RF=ensemble.rf, CUBIST=ensemble.cubist, GBM=ensemble.gbm))
summary(ensembles.output)




# Plot tunning results
dotplot(ensembles.output)




# Detailed review of Cubist
print(ensemble.cubist)



# Plot tunned Cubist
plot(cubist.tuned)


# Data preparation
set.seed(7)
x <- DTset[,1:13]
y <- DTset[,14]
prep.Parms <- preProcess(x, method=c("BoxCox"))
transformX <- predict(prep.Parms, x)
# training Cubist Model
cubist.model <- cubist(x=transformX, y=y, committees=18)


# Transforming validation dataset
set.seed(7)
validation.X <- validateSet[,1:13]
transform.validation.X <- predict(prep.Parms, validation.X)
validation.Y <- validateSet[,14]
# Predictions based on Cubist model with the validation data
predict.output <- predict(cubist.model, newdata=transform.validation.X, neighbors=3)
# Final RMSE
Final.rmse <- RMSE(predict.output, validation.Y)
r_2 <- R2(predict.output, validation.Y)
print(Final.rmse)




