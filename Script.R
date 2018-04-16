
library(dplyr)
library(ggplot2)
library(caret)

#Importing data
df_train <- read.csv("train_ajEneEa.csv")
df_test  <- read.csv("test_v2akXPA.csv")

testids <- df_test$id
df_train$id <- NULL
df_test$id <- NULL

#mean impute missing bmi
#impute bmi with mean
df_train$bmi[which(is.na(df_train$bmi))]<-mean(df_train$bmi,na.rm = T)#tried median too, no better
df_test$bmi[which(is.na(df_test$bmi))]<-mean(df_test$bmi,na.rm = T)

levels(df_train$smoking_status)[levels(df_train$smoking_status)==""] <- "Hesitant"
levels(df_test$smoking_status)[levels(df_test$smoking_status)==""] <- "Hesitant" 

#convert stroke to factor, important for gbm to work in caret
df_train$stroke <- as.factor(df_train$stroke)
levels(df_train$stroke)[levels(df_train$stroke)=="0"]<-"No"
levels(df_train$stroke)[levels(df_train$stroke)=="1"]<-"Yes"

#weights
model_weights <- ifelse(df_train$stroke=="Yes",100/(table(df_train$stroke)[2]),100/(table(df_train$stroke)[1]))

#reusable folds for comparison
myFolds <- createFolds(df_train$stroke, k = 10)

#control and train
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # Important for twoclasssummary
  verboseIter = TRUE,
  savePredictions = TRUE,
  index=myFolds
)
#weghts_model
gbm_model_weights <- train(stroke~.,
                    df_train,
                    method="gbm",
                    weights = model_weights,
                    trControl=myControl,
                    metric="ROC"
                    )
#downsampling
myControl$sampling="down"
gbm_model_down <- train(stroke~.,
                          df_train,
                          method="gbm",
                          trControl=myControl,
                          metric="ROC"
)

#upsampling
myControl$sampling="up"
gbm_model_up <- train(stroke~.,
                        df_train,
                        method="gbm",
                        trControl=myControl,
                        metric="ROC"
)

#smote
myControl$sampling="smote"
gbm_model_smote <- train(stroke~.,
                        df_train,
                        method="gbm",
                        trControl=myControl,
                        metric="ROC"
)

#comparing models
model_list <- list(item1=gbm_model_weights,item2=gbm_model_down,item3=gbm_model_up,item4=gbm_model_smote)
resamples <- resamples(model_list)
summary(resamples)

#gbm_model_up performed the best
test_stroke_up <- predict(gbm_model_up,df_test,type = "prob")
test_stroke_down <-predict(gbm_model_down,df_test,type = "prob")
test_stroke_weights <-predict(gbm_model_weights,df_test,type = "prob")
test_stroke_smote <-predict(gbm_model_smote,df_test,type = "prob")

#Generating output file
test_stroke_up$No <- NULL
test_stroke_down$No <- NULL
test_stroke_weights$No <- NULL
test_stroke_smote$No <- NULL
names(test_stroke_up) <- "stroke"
names(test_stroke_down) <- "stroke"
names(test_stroke_weights) <- "stroke"
names(test_stroke_smote) <- "stroke"
sample_submission_up <-data.frame("id"=testids,"stroke"=test_stroke_up)
sample_submission_down <-data.frame("id"=testids,"stroke"=test_stroke_down)
sample_submission_weights <-data.frame("id"=testids,"stroke"=test_stroke_weights)
sample_submission_smote <-data.frame("id"=testids,"stroke"=test_stroke_smote)
write.csv(sample_submission_up,"sample_submission_up.csv",row.names = F)
write.csv(sample_submission_down,"sample_submission_down.csv",row.names = F)
write.csv(sample_submission_weights,"sample_submission_weights.csv",row.names = F)
write.csv(sample_submission_smote,"sample_submission_smote.csv",row.names = F)

#Which means I need to impute the missing values in a better way, maybe try KNN imputation
