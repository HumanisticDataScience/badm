#In this demo, we will use the German credit set data to build three classifiers
#and compare their performance using a variety of metrics we covered earlier.
#Go here to understand Caret https://topepo.github.io/caret/index.html
library(ROCR) #orphaned package! but does lift and gains
library(pROC)
library(tidyverse)
library(caret)

#Load our dataset
data(GermanCredit)
set.seed(123)
#we will use 70% to train (resamples) and 30% to test (generalizability)
#note: caret uses stratified sampling to construct similar train/test outcome proportions
#for numeric outcomes it create groups based on percentiles and samples from those groups
Train <- createDataPartition(GermanCredit$Class, p=0.7, list=FALSE)
training <- GermanCredit[ Train, ]
testing <- GermanCredit[ -Train, ]

#What is our base rate? Will help us to understand Gains/lift
#What is our positive class? i.e., which class are we interested in predicting?
prop.table(table(GermanCredit$Class))
prop.table(table(training$Class))
prop.table(table(testing$Class))

#Learn a logistic regression model to predict Class
###################################
#Define our resampling scheme
# 5-fold CV x 5 replicates
fitControl <- trainControl(
  method = "repeatedcv", #can use k-folds, repeated k-folds, or bootstrap
  number = 5, #this will do 5-fold cv
  ## repeated ten times
  repeats = 5, #5 replicates of 5-fold cv
  returnResamp = 'final', #return individual predictions
  summaryFunction = twoClassSummary, #for binary outcomes: AUC, Sens, Spec
  classProbs = TRUE)
#######################
#Use train() to actually fit a model using the training set
#we pass in our trainControl resampling object and set metric to ROC (AUC)
mod_logistic <- train(Class ~ .,  data=training, method="glm", family="binomial",
                 trControl=fitControl, metric='ROC')

#Review model fitting and CV
mod_logistic

#See resampling results for each replicate. Result is average for a replicate
mod_logistic$resample

#if you want to calculate your own Confusion matrix metrics
mod_logistic$resampledCM

#see best results. Can use SD to construct 95% CIs +- 2 std errors.
mod_logistic$results

#Now compare with a random forest
#set up resampling procedure
rftrControl= trainControl(
  method = "repeatedcv", #can use k-folds, repeated k-folds, or bootstrap
  number = 5, #this will do 5-fold cv
  ## repeated ten times
  repeats = 5, #10 replicates of 5-fold cv
  returnResamp = 'final',
  summaryFunction = twoClassSummary, #for binary outcomes: AUC, Sens, Spec
  classProbs = TRUE)

#define a tunegrid for number of random selection of predictors to test at each node split
rf_grid = expand.grid(mtry=c(2))

#beware, RF can take a long time with cross-validation. It's fitting different 
#models with different hyperparams in each replicate
rf_default <- train(Class~., 
                    data=training, 
                    method='rf', 
                    metric='ROC', 
                    importance=TRUE, trControl=rftrControl, tuneGrid=rf_grid,
                    verbose=TRUE)

rf_default
#See resampling results for each replicate. Result is average for a replicate
rf_default$resample

#if you want to calculate your own Confusion matrix metrics
rf_default$resampledCM

#see best results
rf_default$results

#Set up LASSO training control
lassoControl= trainControl(
  method = "repeatedcv", #can use k-folds, repeated k-folds, or bootstrap
  number = 5, #this will do 5-fold cv
  ## repeated ten times
  repeats = 5, #10 replicates of 5-fold cv
  returnResamp = 'final',
  summaryFunction = twoClassSummary, #for binary outcomes: AUC, Sens, Spec
  classProbs = TRUE)

#train model
lass_mod <- train(Class ~ .,  data=training, method="glmnet",
      trControl=lassoControl, metric='ROC')

#which model parameters were best according to our cross validation?
lass_mod

################# GENERATE PREDICTIONS
#Normally, we would choose the best of three and then look at generalizability for
#this final model. We will cheat slightly and look at performance on test set for all 3.
#Estimating Performance
class_pred <- predict(mod_logistic, newdata=testing) #classes using default .50 cutoff
predict(mod_logistic, newdata=testing, type="prob") #numeric probs. we can set cutoff
#multicollinearity issues... but we would re
findLinearCombos() could help

#we want probabilities to calculate AUC and Gains/lift (ordering in desc)
logistic_preds <- predict(mod_logistic, newdata=testing, type="prob") 

#Get preds from RF
rf_preds <- predict(rf_default, newdata=testing, type='prob')

#get preds from LASSO
lasso_preds <- predict(lass_mod, newdata=testing, type='prob')


#Combine all results into one big dataframe to make easier to plot later
#We are only interested in Class= "Good" (our positive class)
full_preds <- data.frame(log_good = logistic_preds$Good,
                         rf_good = rf_preds$Good,
                         lass_good= lasso_preds$Good,
                         actuals = testing$Class)

#########################################################
#FINDING OUT AVERAGE COST OF PREDICTION
#you could set up your own costs if you wanted to
#costs Negative if bad and say good (FP), compared to (FN): Benefits Positive
#it's good but we say bad is 10x worse than converse. 
costMatrix <- matrix(c(1,-2,-10,5), nrow=2, byrow = T) #TN: 0, FN: -2, FP: -10, TP:5
costMatrix
#check baseline confusion matrix at .5 cutoff
#be sure to check Positive Class!
cm <- confusionMatrix(data=class_pred, testing$Class, positive = 'Good')
cm

#AVERAGE COST PER PREDICTION
#look at costs divided by number of predictions to get average cost per prediction.
number_preds <- length(testing$Class)

#using this classifier, we expect to make about $1.15 per prediction
#what happens if the cost of a FP changes to -$100?
sum(cm$table * costMatrix)/number_preds

###########################################################
#EVALUATING PREDICTIVE VALIDITY: DISCRIMINATION: ROC/AUC
#Remind ourselves of AUC interpretation
library(ROCR)

# Compute AUC for predicting Class with the model
#get probabilities of 2 class for logistic model
prob_logistic <- predict(mod_logistic, newdata=testing, type="prob")

#create prediction Object by passing in Pred Prob of Good (positive class) and Actuals
pred <- prediction(prob_logistic$Good, testing$Class)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
#normally you would compare multiple classifiers here. 
#Q: what does it mean if ROC curves cross? 

#To find the AUC score corresponding to this ROC curve
#remember, AUC is an AGGREGATE measure like the mean. A higher AUC does not mean
#uniformly better classifier. Why not?
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

#to find Precision-Recall performance: Precision on Y axis, Recall on X axis
#what does this tell us? 
gain_germlog <- performance(pred, measure = 'prec', 'rec')
plot(gain_germlog, main='Precision-Recall LogReg')

###################################
#EVALUATING CONSTRAINED RANKING: GAINS AND LIFT PLOTS

#add naive prediction based on training set for comparison
#now you can see where the benchmark comes from
#at 70% samples tested, we expect to find 70% of TPs
full_preds$naive <- 0.7
#in best case, we always rank TP above TN until run out of TPs in test set
full_preds$best_case <- ifelse(full_preds$actuals == 'Good', .99, .01)

#Cumulative Gains Chart
#you can add multiple model predictions such as "logreg" or "RF"
lft_obj <- lift(actuals ~ rf_good + lass_good + log_good + naive + best_case, data=full_preds, class='Good')

#we can pass in the lift object to ggplot to customize it
#Compare with model that randomly selects with prob = base rate
ggplot(lft_obj)+
  theme_minimal()+
  labs(title='Cumulative Gains Chart: Predicting Good Credit')

#Look at your base rate to determine upper limit scenario of best gains
#Best case: your classifier ranks all TPs above TNs
prop.table(table(full_preds$actuals))

#LIFT CHART: Derived from our Gains Chart. 
#Slightly more complicated to make decile lift. 
#
library(gains)
#need to convert ot 1/0s first not factors
#convert from factors to numeric for the gains() function
full_preds$num_actuals <- ifelse(full_preds$actuals== 'Good', 1, 0)

#get results
#groups = 10 means cut into 10 bins (deciles)
my_g <- gains(actual=full_preds$num_actuals, predicted= full_preds$log_good, groups=10, 
              percents=TRUE, optimal=TRUE)
my_g

#you can plot this directly but much nicer to use ggplot
plot(my_g)
#compare with naive
naive_g <- gains(actual = full_preds$num_actuals, predicted = full_preds$naive,
                 groups=10, percents = TRUE, optimal=TRUE )
naive_g
#compute deciles for plotting. Note, since all same pred prob, can't break into deciles!
naive_df <- data.frame(mean_resp = naive_g$mean.resp/mean(full_preds$num_actuals), deciles = naive_g$depth)

#This shows you the benchmark lift of 1: % TPs found = % samples scanned (ratio = 1)
ggplot(naive_df, aes(deciles, mean_resp))+
  geom_col()+
  scale_x_continuous(breaks = my_g$depth)+
  scale_y_continuous(breaks = seq(0,max(res_df$mean_resp),.1))+
  geom_hline(yintercept=1, color='red', linetype='dotted')+
  theme_minimal()

#Extract the decile information from gains() chart we computed earlier.
res_df <- data.frame(mean_resp = my_g$mean.resp/mean(full_preds$num_actuals), deciles = my_g$depth)

#plot the decile lift chart. 
#What's the interpretation of this?
ggplot(res_df, aes(deciles, mean_resp))+
  geom_col()+
  scale_x_continuous(breaks = my_g$depth)+
  scale_y_continuous(breaks = seq(0,max(res_df$mean_resp),.1))+
  geom_hline(yintercept=1, color='red', linetype='dotted')+
  theme_minimal()+
  labs(title='Decile Lift Chart: Logistic Regression', y = 'Mean Response', x='Decile')

################################################################
#EVALUATING PREDICTIVE VALIDITY: CALIBRATION. DOUBLE DENSITY PLOT
#our goal is maximal separation between the two density plots
ggplot(full_preds, aes(rf_good, fill=actuals))+
  geom_density(alpha=.5)

#compare multiple
library(tidyr)
full_preds%>%
  gather(-actuals, key='var', value ='val')%>%
  dplyr::filter(var %in% c('lass_good', 'rf_good', 'log_good'))%>%
  ggplot(aes(val, fill=actuals))+
  geom_density(alpha=.5)+
  facet_wrap(~var, ncol=1)+
  labs(title='Double Density Plot: Classifier Calibration', x='Predicted Prob of Good')

##############################################################
#Extra time? Use the datasets::iris to classify setosas
#To get you started for binary classification
df_iris <- datasets::iris
df_iris$Species <- ifelse(df_iris$Species == 'setosa', 'setosa', 'non-setosa')

#change to binary classification (all non-setosa == non-setosa label)
#set.seed(123)
#use 70% for training and 30% testing
#build a logistic regresion to predict setosas

# EVALUATE:
# Gains/lift curve
# Cost matrix (you decide classification costs)
# AUC/ROC

