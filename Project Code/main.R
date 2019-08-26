################# Niall O'Neill TFM #########################
##### Prediction Models for ECGs using Ensemble Methods ##### 
#############################################################


################ Set up Environment #########################
setwd("~/Big Data Masters/Trabajo Fin De Masters/Project Code")
wrk_dir <- "C:/Users/niall/OneDrive/Documents/Big Data Masters/Trabajo Fin De Masters/"

#First load the signal library 
library("signal")
library("zoo")
library("pracma")
library("Rwave")
library("warbleR")
library("caret")
library("randomForest")
library("mlbench")
library("e1071")
library("fastAdaboost")
library("xgboost")
library("adabag")
library("pROC")
library("DMwR")
library("dplyr")
library("kernlab")
library("skimr")
library("ggplot2")
library("corrplot")


############## Source #######################################

source("preprocess_signal.R")
source("generate_params_df.R")
source("calculate_section_parameters.R")
source("tci.R")
source("mav.R")
source("calculate_counts.R")
source("calculate_exp.R")
source("vfleak.R")
source("kolmogorov.R")
source("cm_jekova.R")
source("peakdet.R")

########### Feature Extraction from ECG files ###############

#Set array of folders names to loop through
folders <- c("CUDB Data/","VFDB Data/")
sampling_rate <- c(250,360)

#Shockable Labels 
shockable <- c("(VF", "[", "]", "(VT", "(VFL", "(VFIB")

all_data <- data.frame(tci = double(), mav = double(), count1 =double(), count2 = double(),
                   count3 = double(), exp = double(), vfleak = double(),
                   cm = double(), cvbin = double(), frqbin = double(), 
                   abin = double(), kurt = double(), label = character(), stringsAsFactors = FALSE)


for (i in 1:length(folders)){
  files <- list.files(path = paste(wrk_dir,folders[i],'Data/', sep =""), pattern ="*.csv", full.names=FALSE, recursive = FALSE)
  samp_rate <- sampling_rate[i]
  
  for (file in files){
    preproc_data <- preprocess_signal(file)
    processed_file <- generate_params_df(preproc_data)
    all_data <- rbind(all_data,processed_file)
    assign(substr(file,1,  nchar(file)-4), processed_file)
    
  }
}

############### Parallelize the environment #################
library(doParallel)
numberofcores = detectCores() 

cl <- makePSOCKcluster(numberofcores)
registerDoParallel(cl)

############### Prediction and Confusion Matrix #############
predictandCM <- function(model,data)
{
  pred <-predict(model,data,type="raw")
  confusionMatrix(pred, reference=testData$label, positive = "Sh")
}

############### Descriptive Analysis ########################

vfdb_data <- bind_rows(list(`418`,`419`,`420`,`421`,`422`,`423`,
                            `424`,`425`, `426`,`427`, `428`, `429`,
                            `430`, `602`, `605`, `607`,`609`, `610`,
                            `611`, `612`, `614`, `615`))

cu_data <- bind_rows(list(cu01,cu02,cu03,cu04,cu05,cu06,cu07,cu08,
                          cu09,cu10,cu11,cu12,cu13,cu14,cu15,cu16,
                          cu17,cu18,cu19,cu20,cu21,cu22,cu23,cu24,
                          cu25,cu26,cu27,cu28,cu29,cu30,cu31,cu32,
                          cu33,cu34,cu35))

table(cu_data$label)
# NSh   Sh 
# 3291  880 

table(vfdb_data$label)
# NSh   Sh 
# 5651 1229

table(all_data$label)
#  NSh   Sh 
#  8942 2109 

skimmed_all <- skim_to_wide(all_data)

nsh_data <- all_data[all_data$label == "NSh",]
sh_data <- all_data[all_data$label == "Sh",]

skimmed_nsh <- skim_to_wide(nsh_data)
skimmed_sh <- skim_to_wide(sh_data)

## TCI 
tci_hist <- ggplot(data = all_data, aes(x=all_data$tci, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "TCI", x=" ") + theme_bw() +
  theme(legend.position = c(0.85, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
tci_hist

## MAV
mav_hist <- ggplot(data = all_data, aes(x=all_data$mav, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "MAV", x=" ") + theme_bw() +
  theme(legend.position = c(0.85, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
mav_hist

## Count 1
count1_hist <- ggplot(data = all_data, aes(x=all_data$count1, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "Count 1", x=" ") + theme_bw() +
  theme(legend.position = c(0.85, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
count1_hist

## Count 2
count2_hist <- ggplot(data = all_data, aes(x=all_data$count2, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "Count 2", x=" ") + theme_bw() +
  theme(legend.position = c(0.1, 0.8), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
count2_hist

## Count 3
count3_hist <- ggplot(data = all_data, aes(x=all_data$count3, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "Count 3", x=" ") + theme_bw() +
  theme(legend.position = c(0.85, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
count3_hist

## Exp
exp_hist <- ggplot(data = all_data, aes(x=all_data$exp, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5,show.legend = FALSE) +
  labs(title = "Exp", x=" ") + theme_bw() +
  theme(legend.position = c(0.85, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
exp_hist

## vfleak
vfleak_hist <- ggplot(data = all_data, aes(x=all_data$vfleak, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "VFleak", x=" ") + theme_bw() +
  theme(legend.position = c(0.15, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
vfleak_hist

## cm
cm_hist <- ggplot(data = all_data, aes(x=all_data$cm, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "Complexity Measure", x=" ") + theme_bw() +
  theme(legend.position = c(0.85, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
cm_hist

## cvbin
cvbin_hist <- ggplot(data = all_data, aes(x=all_data$cvbin, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "Covariance", x=" ") + theme_bw() +
  theme(legend.position = c(0.85, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
cvbin_hist

## frqbin
frqbin_hist <- ggplot(data = all_data, aes(x=all_data$frqbin, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "Frequency", x=" ") + theme_bw() +
  theme(legend.position = c(0.85, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
frqbin_hist

## abin
abin_hist <- ggplot(data = all_data, aes(x=all_data$abin, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5, show.legend = FALSE) +
  labs(title = "Area", x=" ") + theme_bw() +
  theme(legend.position = c(0.85, 0.85), plot.title = element_text(hjust = 0.5)) +
  guides(fill=guide_legend(title=" ")) 
abin_hist

## kurt
kurt_hist <- ggplot(data = all_data, aes(x=all_data$kurt, fill = all_data$label)) +
  geom_histogram(bins = 30, alpha = 0.5) +
  labs(title = "Kurtosis", x=" ") + theme_bw() +
  theme(legend.position = c(0.8, 0.6), plot.title = element_text(hjust = 0.5),legend.title=element_blank()) 
kurt_hist

## Grid Arrangement
grid.arrange(tci_hist,mav_hist,count1_hist,count2_hist,
             count3_hist,exp_hist, vfleak_hist,cm_hist,
             cvbin_hist, frqbin_hist, abin_hist, kurt_hist,
             nrow = 3, ncol = 4)


## Correlation 
correlations <- cor(all_data[,1:12])
correlations
corr_plot <-corrplot(correlations, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


#             tci         mav     count1      count2      count3         exp      vfleak          cm       cvbin     frqbin        abin       kurt
# tci     1.00000000  0.09648547 -0.3523629 -0.16224082 -0.37043104 -0.23579526  0.29040087 -0.24433375 -0.35654637 -0.6143578  0.19224540  0.3320385
# mav     0.09648547  1.00000000  0.2612911 -0.18228705  0.27570019  0.20604834 -0.29486120  0.04704714  0.11379811 -0.1823987 -0.16537497 -0.1768941
# count1 -0.35236294  0.26129114  1.0000000  0.12667503  0.99051632  0.34565441 -0.63280337  0.16746684  0.26694694  0.3706884 -0.13081683 -0.3360790
# count2 -0.16224082 -0.18228705  0.1266750  1.00000000  0.08654707  0.15505000  0.05889349  0.04661461 -0.02517235  0.2159049  0.60183652 -0.0501725
# count3 -0.37043104  0.27570019  0.9905163  0.08654707  1.00000000  0.34610961 -0.66447302  0.17408844  0.28009146  0.3899049 -0.17701716 -0.3400996
# exp    -0.23579526  0.20604834  0.3456544  0.15505000  0.34610961  1.00000000 -0.27290067  0.14842952  0.23579922  0.2381652 -0.07640868 -0.2926360
# vfleak  0.29040087 -0.29486120 -0.6328034  0.05889349 -0.66447302 -0.27290067  1.00000000 -0.22624912 -0.37882869 -0.2557085  0.35499548  0.4493494
# cm     -0.24433375  0.04704714  0.1674668  0.04661461  0.17408844  0.14842952 -0.22624912  1.00000000  0.66953022  0.3407630 -0.48154787 -0.3937462
# cvbin  -0.35654637  0.11379811  0.2669469 -0.02517235  0.28009146  0.23579922 -0.37882869  0.66953022  1.00000000  0.4724604 -0.77430325 -0.6200795
# frqbin -0.61435785 -0.18239868  0.3706884  0.21590488  0.38990495  0.23816521 -0.25570848  0.34076295  0.47246043  1.0000000 -0.27309568 -0.3023760
# abin    0.19224540 -0.16537497 -0.1308168  0.60183652 -0.17701716 -0.07640868  0.35499548 -0.48154787 -0.77430325 -0.2730957  1.00000000  0.4146925
# kurt    0.33203846 -0.17689407 -0.3360790 -0.05017250 -0.34009956 -0.29263602  0.44934938 -0.39374624 -0.62007949 -0.3023760  0.41469249  1.0000000


############### Split data into train and test sets #########
set.seed(100)
all_data$label <- as.factor(all_data$label)
train_indices <- createDataPartition(all_data$label, p=0.8, list = FALSE)


#Create training and test datasets 
trainData <- all_data[train_indices,]
testData <- all_data[-train_indices,]
testDatax <- testData[,1:12]

############### Baseline Model ###############################

##### Random Forest ##########################################

fitControl_grid <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3, classProbs = TRUE,
                           search = 'grid')

set.seed(100)

tunegrid_rf <- expand.grid(mtry=c(1:sqrt(ncol(trainData)-1)))
start_time <- Sys.time()
random_forest <- train(label ~ ., 
                       data = trainData,
                       method = 'rf',
                       metric = 'Accuracy',
                       tuneGrid = tunegrid_rf,
                       preProc = c("center","scale"),
                       trControl = fitControl_grid)
end_time <- Sys.time()

rfTime <- end_time - start_time
## RF Training time  = 2.304861 mins

predictandCM(random_forest,testDatax)

######## Output #################

# Confusion Matrix and Statistics

# Reference
# Prediction  NSh   Sh
# NSh 1751   44
# Sh    37  377

# Accuracy : 0.9633          
# 95% CI : (0.9546, 0.9708)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.8804          

# Mcnemar's Test P-Value : 0.505           
#                                           
#             Sensitivity : 0.8955          
#             Specificity : 0.9793         
#          Pos Pred Value : 0.9106         
#          Neg Pred Value : 0.9755          
#              Prevalence : 0.1906          
#          Detection Rate : 0.2073          
#    Detection Prevalence : 0.1874          
#       Balanced Accuracy : 0.9374          
#                                           
#        'Positive' Class : Sh 

random_forest$finalModel$importance

# Random Forest Feature Importance
#             MeanDecreaseGini
# tci           215.74718
# mav           208.47640
# count1        354.05206
# count2         71.92368
# count3        528.13707
# exp            46.77664
# vfleak        669.61718
# cm             53.75769
# cvbin         155.98656
# frqbin        137.64457
# abin          107.57472
# kurt          182.10423

############# AdaBoost #####################################

tunegrid_adaBoost <- expand.grid(mtry=c(1:ncol(trainData)-1))

set.seed(100)

start_time <- Sys.time()

adaboost <- train(label ~ ., 
                  data = trainData,
                  method = 'adaboost',
                  metric = 'Accuracy',
                  preProc = c("center","scale"),
                  trControl = fitControl_grid)

end_time <- Sys.time()

adaboostTime <- end_time - start_time
## Adaboost Training Time = 27.89922 mins

predictandCM(adaboost,testDatax)

################# Output ########
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1753   37
# Sh           35  384
# 
# Accuracy : 0.9674          
# 95% CI : (0.9591, 0.9744)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.8942          
# 
# Mcnemar's Test P-Value : 0.9062          
#                                           
#             Sensitivity : 0.9121          
#             Specificity : 0.9804          
#          Pos Pred Value : 0.9165          
#          Neg Pred Value : 0.9793          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1738          
#    Detection Prevalence : 0.1897          
#       Balanced Accuracy : 0.9463          
#                                           
#        'Positive' Class : Sh    

set.seed(100)

start_time <- Sys.time()
adaboost.m1 <- train(label ~ ., 
                     data = trainData,
                     method = 'AdaBoost.M1',
                     metric = 'Accuracy',
                     preProc = c("center","scale"),
                     trControl = fitControl_grid)

end_time <- Sys.time()

adaboostm1Time <- end_time - start_time
# Adaboostm1 Training time = 1.085028 hr = 65.10168 mins

predictandCM(adaboost.m1,testDatax)

############Output ###########
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1758   38
# Sh           30  383
# 
# Accuracy : 0.9692         
# 95% CI : (0.9611, 0.976)
# No Information Rate : 0.8094         
# P-Value [Acc > NIR] : <2e-16         
# 
# Kappa : 0.8995         
# 
# Mcnemar's Test P-Value : 0.396          
#                                          
#             Sensitivity : 0.9097         
#             Specificity : 0.9832         
#          Pos Pred Value : 0.9274         
#          Neg Pred Value : 0.9788         
#              Prevalence : 0.1906         
#          Detection Rate : 0.1734         
#    Detection Prevalence : 0.1870         
#       Balanced Accuracy : 0.9465         
#                                          
#        'Positive' Class : Sh 

#Importance Plot 
importanceplot(adaboost.m1$finalModel)

############### XGBTree ######################################

fitControl_random <- trainControl(method = "repeatedcv",
                                number = 10,
                                repeats = 3, classProbs = TRUE,
                                search = 'random')


set.seed(100)

start_time <- Sys.time()
xgb <- train(label ~ ., 
             data = trainData,
             method = 'xgbTree',
             metric = 'Accuracy',
             preProc = c("center","scale"),
             trControl = fitControl_grid)

end_time <- Sys.time()

xgbTime <- end_time - start_time
# XGB Training Time = 6.0623603 mins

predictandCM(xgb,testDatax)

################## Output #######

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1754   40
# Sh           34  381
# 
# Accuracy : 0.9665          
# 95% CI : (0.9581, 0.9736)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.8908          
# 
# Mcnemar's Test P-Value : 0.5611          
#                                           
#             Sensitivity : 0.9050          
#             Specificity : 0.9810          
#          Pos Pred Value : 0.9181          
#          Neg Pred Value : 0.9777          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1725          
#    Detection Prevalence : 0.1879          
#       Balanced Accuracy : 0.9430          
#                                           
#        'Positive' Class : Sh

# XGBoost Feature Importance 
xgb_importance <- xgb.importance(feature_names = xgb$finalModel$feature_names,
                                 model = xgb$finalModel)

#    Feature        Gain      Cover  Frequency
# 1:  count3 0.309942717 0.13971989 0.11837655
# 2:  vfleak 0.274363847 0.15079187 0.12175874
# 3:     mav 0.121862874 0.18016828 0.17361894
# 4:   cvbin 0.066557998 0.05727257 0.05862458
# 5:  frqbin 0.061449335 0.06644735 0.06538895
# 6:     tci 0.044871688 0.06634496 0.08455468
# 7:  count2 0.032210855 0.04537118 0.05975197
# 8:    kurt 0.024768137 0.08382589 0.07553551
# 9:  count1 0.022474578 0.07431545 0.08229989
# 10:    abin 0.021887453 0.05698350 0.06200676
# 11:     exp 0.011901600 0.03727114 0.04058625
# 12:      cm 0.007708918 0.04148791 0.05749718


#################### Adabag ##################################

set.seed(100)
start_time <- Sys.time()
adabag <- train(label ~ ., 
             data = trainData,
             method = 'AdaBag',
             metric = 'Accuracy',
             preProc = c("center","scale"),
             trControl = fitControl_grid)
end_time <- Sys.time()

adabagTime <- end_time - start_time
# Adabag Training Time = 19.31805 mins

predictandCM(adabag,testDatax)

############# Output ############

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1715   52
# Sh           73  369
# 
# Accuracy : 0.9434          
# 95% CI : (0.9329, 0.9527)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2e-16         
# 
# Kappa : 0.82            
# 
# Mcnemar's Test P-Value : 0.07364         
#                                           
#             Sensitivity : 0.8765          
#             Specificity : 0.9592          
#          Pos Pred Value : 0.8348          
#          Neg Pred Value : 0.9706          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1670          
#    Detection Prevalence : 0.2001          
#       Balanced Accuracy : 0.9178          
#                                           
#        'Positive' Class : Sh  

# Adabag Feature Importance 
adabag$finalModel$importance

#      abin          cm      count1      count2      count3       cvbin     
# 0.77190013  0.00000000  0.08516284  0.00000000 16.99487766  3.65420604 

#     exp      frqbin        kurt         mav         tci      vfleak 
# 0.00000000  1.51251477  0.04398820  2.92578310 13.50871826 60.50284901


# Plot 
importanceplot(adabag$finalModel)

################## SMOTE Sampling  #############################

smote_train <- SMOTE(label ~ ., data  = trainData)
set.seed(100)

start_time <- Sys.time()
smote_rf <- train(label ~ ., 
                       data = smote_train,
                       method = 'rf',
                       metric = 'Accuracy',
                       tuneGrid = tunegrid_rf,
                       preProc = c("center","scale"),
                       trControl = fitControl_grid)

end_time <- Sys.time()

smote_rfTime <- end_time - start_time
# Smote RF Training Time = 2.515431 mins

predictandCM(smote_rf,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1734   23
# Sh           54  398
# 
# Accuracy : 0.9651          
# 95% CI : (0.9566, 0.9724)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8901          
# 
# Mcnemar's Test P-Value : 0.0006289       
#                                           
#             Sensitivity : 0.9454          
#             Specificity : 0.9698          
#          Pos Pred Value : 0.8805          
#          Neg Pred Value : 0.9869          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1802          
#    Detection Prevalence : 0.2046          
#       Balanced Accuracy : 0.9576          
#                                           
#        'Positive' Class : Sh


# Smote Random Forest Feature Importance 
#            MeanDecreaseGini
# tci            422.6500
# mav            432.8136
# count1         649.7552
# count2         161.4268
# count3         973.4128
# exp            154.7816
# vfleak        1343.7350
# cm             152.6989
# cvbin          360.9955
# frqbin         333.0339
# abin           275.7742
# kurt           525.8561

set.seed(100)

start_time <- Sys.time()
smote_adaboost <- train(label ~ ., 
                  data = smote_train,
                  method = 'adaboost',
                  metric = 'Accuracy',
                  preProc = c("center","scale"),
                  trControl = fitControl_grid)

end_time <- Sys.time()

smote_adaboostTime <- end_time - start_time
#Smote Adaboost Training Time = 46.39612 mins

predictandCM(smote_adaboost,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1744   17
# Sh           44  404
# 
# Accuracy : 0.9724          
# 95% CI : (0.9647, 0.9788)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.9126          
# 
# Mcnemar's Test P-Value : 0.0008717       
#                                           
#             Sensitivity : 0.9596          
#             Specificity : 0.9754          
#          Pos Pred Value : 0.9018          
#          Neg Pred Value : 0.9903          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1829          
#    Detection Prevalence : 0.2028          
#       Balanced Accuracy : 0.9675          
#                                           
#        'Positive' Class : Sh


# Smote Adaboost.M1 
set.seed(100)

start_time <- Sys.time()
smote_adaboost.m1 <- train(label ~ ., 
                           data = smote_train,
                           method = 'AdaBoost.M1',
                           metric = 'Accuracy',
                           preProc = c("center","scale"),
                           trControl = fitControl_grid)
end_time <- Sys.time()

smote_adaboostm1Time <- end_time - start_time
# Smote Adaboostm1 Training Time = 59.79215 mins 

predictandCM(smote_adaboost.m1,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1719   27
# Sh           69  394
# 
# Accuracy : 0.9565          
# 95% CI : (0.9472, 0.9647)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8643          
# 
# Mcnemar's Test P-Value : 2.857e-05       
#                                           
#             Sensitivity : 0.9359          
#             Specificity : 0.9614          
#          Pos Pred Value : 0.8510          
#          Neg Pred Value : 0.9845          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1784          
#    Detection Prevalence : 0.2096          
#       Balanced Accuracy : 0.9486          
#                                           
#        'Positive' Class : Sh  

importanceplot(smote_adaboost.m1$finalModel)

set.seed(100)

start_time <- Sys.time()
smote_xgb <- train(label ~ ., 
             data = smote_train,
             method = 'xgbTree',
             metric = 'Accuracy',
             preProc = c("center","scale"),
             trControl = fitControl_grid)
end_time <- Sys.time()

smote_xgbTime <- end_time - start_time
#Smote xgb training time = 5.357322 mins

predictandCM(smote_xgb,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1732   25
# Sh           56  396
# 
# Accuracy : 0.9633          
# 95% CI : (0.9546, 0.9708)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8844          
# 
# Mcnemar's Test P-Value : 0.0008581       
#                                           
#             Sensitivity : 0.9406          
#             Specificity : 0.9687          
#          Pos Pred Value : 0.8761          
#          Neg Pred Value : 0.9858          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1793          
#    Detection Prevalence : 0.2046          
#       Balanced Accuracy : 0.9546          
#                                           
#        'Positive' Class : Sh

# Smote XGBoost Feature Importance
# XGBoost Feature Importance 
smote_xgb_importance <- xgb.importance(feature_names = smote_xgb$finalModel$feature_names,
                                 model = smote_xgb$finalModel)

#    Feature        Gain      Cover  Frequency
# 1:  count3 0.338686735 0.13409033 0.10741139
# 2:  vfleak 0.271058007 0.12873373 0.10633727
# 3:     mav 0.111610613 0.18588957 0.16756176
# 4:   cvbin 0.067421030 0.06218600 0.05155747
# 5:  frqbin 0.043182255 0.09527376 0.09667025
# 6:     tci 0.041713701 0.07354897 0.07626208
# 7:     exp 0.031208302 0.07908182 0.06229860
# 8:    kurt 0.028802686 0.05964381 0.07518797
# 9:    abin 0.028505988 0.04705384 0.06122449
# 10:  count2 0.017228084 0.04180667 0.06659506
# 11:  count1 0.015435319 0.06111288 0.08592911
# 12:      cm 0.005147282 0.03157862 0.04296455

set.seed(100)
start_time <- Sys.time()
smote_adabag <- train(label ~ ., 
                data = smote_train,
                method = 'AdaBag',
                metric = 'Accuracy',
                preProc = c("center","scale"),
                trControl = fitControl_grid)
end_time <- Sys.time()

smote_adabagTime <- end_time - start_time
# Smote adabag training time = 20.24596 mins

predictandCM(smote_adabag,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1612   26
# Sh          176  395
# 
# Accuracy : 0.9086          
# 95% CI : (0.8958, 0.9203)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7391          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.9382          
#             Specificity : 0.9016          
#          Pos Pred Value : 0.6918          
#          Neg Pred Value : 0.9841          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1788          
#    Detection Prevalence : 0.2585          
#       Balanced Accuracy : 0.9199          
#                                           
#        'Positive' Class : Sh 

# SMOTE Adabag feature importance 
smote_adabag$finalModel$importance

#   abin        cm    count1    count2    count3     cvbin  
# 2.078302  0.000000  0.000000  1.711928  5.317259  0.613209
#     exp     frqbin      kurt       mav       tci    vfleak 
#  0.000000  5.843330  0.000000  5.686386  3.712559 75.037027  

importanceplot(smote_adabag$finalModel)


################### Further Tuning #############################
# After running the basic models it can be seen that the best
# results come from the xgboost model and the adaboost model
# Next these methods are investigated further with Hyperparameter
# Tuning

fitControl_xgbgrid <- trainControl(method = "repeatedcv",
                                number = 10,
                                repeats = 3, classProbs = TRUE,
                                allowParallel = TRUE,
                                search = 'grid')

tunegrid_xgb <- expand.grid(nrounds = c(150,400,600), 
                            max_depth = c(3, 5, 10), 
                            eta = c(0,6, 0.4, 0.1), 
                            gamma = 0, 
                            colsample_bytree = 0.8, 
                            min_child_weight = 1,
                            subsample = c(0.7, 1.0))
set.seed(100)
start_time <- Sys.time()
xgb_tune <- train(label ~ ., 
                  data = trainData,
                  method = 'xgbTree',
                  metric = 'Accuracy',
                  preProc = c("center","scale"),
                  trControl = fitControl_xgbgrid,
                  tuneGrid = tunegrid_xgb)
end_time <- Sys.time()

xgb_tuneTime <- end_time - start_time
# Time difference of 25.05532 mins

predictandCM(xgb_tune,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1756   33
# Sh           32  388
# 
# Accuracy : 0.9706          
# 95% CI : (0.9626, 0.9772)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.9045          
# 
# Mcnemar's Test P-Value : 1               
#                                           
#             Sensitivity : 0.9216          
#             Specificity : 0.9821          
#          Pos Pred Value : 0.9238          
#          Neg Pred Value : 0.9816          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1756          
#    Detection Prevalence : 0.1901          
#       Balanced Accuracy : 0.9519          
#                                           
#        'Positive' Class : Sh  

xgbtune_importance <- xgb.importance(feature_names = xgb_tune$finalModel$feature_names,
                                 model = xgb_tune$finalModel)

#    Feature       Gain      Cover  Frequency
# 1:  vfleak 0.35044061 0.16596447 0.11050940
# 2:  count3 0.18113454 0.12904698 0.10229707
# 3:     mav 0.10270941 0.17865393 0.14722685
# 4:     tci 0.09343515 0.07801008 0.10069031
# 5:  count1 0.05224208 0.07376983 0.07635087
# 6:  frqbin 0.04735132 0.08272762 0.06331826
# 7:   cvbin 0.04435725 0.05166635 0.06308022
# 8:    kurt 0.03649473 0.07573553 0.08545584
# 9:  count2 0.03298386 0.05960176 0.08039752
# 10:    abin 0.02958711 0.03785437 0.05409426
# 11:     exp 0.01500409 0.03923526 0.04736967
# 12:      cm 0.01425985 0.02773382 0.06920971

tunegrid_smotexgb <- expand.grid(nrounds = c(600,800,1000), 
                            max_depth = c(7, 10, 15), 
                            eta =  0.1, 
                            gamma = 0, 
                            colsample_bytree = 0.8, 
                            min_child_weight = 1,
                            subsample = c(0.7, 1.0))
set.seed(100)
start_time <- Sys.time()
smotexgb_tune <- train(label ~ ., 
                  data = smote_train,
                  method = 'xgbTree',
                  metric = 'Accuracy',
                  preProc = c("center","scale"),
                  trControl = fitControl_xgbgrid,
                  tuneGrid = tunegrid_smotexgb)
end_time <- Sys.time()

smotexgb_tuneTime <- end_time - start_time
# Time difference of 22.96349 mins

predictandCM(smotexgb_tune,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1742   20
# Sh           46  401
# 
# Accuracy : 0.9701          
# 95% CI : (0.9621, 0.9768)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.9054          
# 
# Mcnemar's Test P-Value : 0.002089        
#                                           
#             Sensitivity : 0.9525          
#             Specificity : 0.9743          
#          Pos Pred Value : 0.8971          
#          Neg Pred Value : 0.9886          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1815          
#    Detection Prevalence : 0.2024          
#       Balanced Accuracy : 0.9634          
#                                           
#        'Positive' Class : Sh 

smote_xgbtune_importance <- xgb.importance(feature_names = smotexgb_tune$finalModel$feature_names,
                                     model = smotexgb_tune$finalModel)

#    Feature       Gain      Cover  Frequency
# 1:  vfleak 0.31441732 0.15735955 0.10500074
# 2:  count3 0.26484691 0.14924611 0.09404157
# 3:     mav 0.10484547 0.16096179 0.14044528
# 4:     tci 0.06139532 0.09172696 0.09221504
# 5:  frqbin 0.05069624 0.10145212 0.08471146
# 6:   cvbin 0.04000696 0.04412842 0.05662240
# 7:    abin 0.03980541 0.05589187 0.06748285
# 8:  count2 0.03222921 0.06062208 0.07794836
# 9:    kurt 0.03042812 0.05942829 0.07933060
# 10:  count1 0.02605383 0.04006997 0.06684109
# 11:     exp 0.02317758 0.05224075 0.06763094
# 12:      cm 0.01209764 0.02687209 0.06772967

#############################################################

###### SVMs #################################################

set.seed(100)
start_time <- Sys.time()
linear_svm <- train(label ~ ., 
                data = trainData,
                method = 'svmLinear',
                metric = 'Accuracy',
                preProc = c("center","scale"),
                trControl = fitControl_grid)
end_time <- Sys.time()

linSVMTime <- end_time - start_time
# Linear SVM Training Time = 29.61545 seconds = 0.49359 mins

predictandCM(linear_svm,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1744   84
# Sh           44  337
# 
# Accuracy : 0.9421          
# 95% CI : (0.9315, 0.9514)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8051          
# 
# Mcnemar's Test P-Value : 0.0005665       
#                                           
#             Sensitivity : 0.8005          
#             Specificity : 0.9754          
#          Pos Pred Value : 0.8845          
#          Neg Pred Value : 0.9540          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1526          
#    Detection Prevalence : 0.1725          
#       Balanced Accuracy : 0.8879          
#                                           
#        'Positive' Class : Sh         

tunegrid_svm <- expand.grid(C = c(0.5 , 0.75,1, 1.25, 1.5, 1.75, 2))                                

set.seed(100)
start_time <- Sys.time()
radial_svm <- train(label ~ ., 
                    data = trainData,
                    method = 'svmRadial',
                    metric = 'Accuracy',
                    preProc = c("center","scale"),
                    trControl = fitControl_grid)
end_time <- Sys.time()

radialSVMTime <- end_time - start_time
# Radial SVM Training Time = 1.342399 mins 

predictandCM(radial_svm,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1745   46
# Sh           43  375
# 
# Accuracy : 0.9597          
# 95% CI : (0.9507, 0.9675)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.8691          
# 
# Mcnemar's Test P-Value : 0.8321          
#                                           
#             Sensitivity : 0.8907          
#             Specificity : 0.9760          
#          Pos Pred Value : 0.8971          
#          Neg Pred Value : 0.9743          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1698          
#    Detection Prevalence : 0.1892          
#       Balanced Accuracy : 0.9333          
#                                           
#        'Positive' Class : Sh          
 
tunegrid_svm_radial  <- expand.grid(C = c(2, 2.5, 3, 3.5, 4, 5),
                                    sigma = c(0.2, 0.3, 0.4, 0.5))
set.seed(100)
start_time <- Sys.time()
radial_svm_tuned <- train(label ~ ., 
                    data = trainData,
                    method = 'svmRadial',
                    metric = 'Accuracy',
                    preProc = c("center","scale"),
                    tuneGrid = tunegrid_svm_radial,
                    trControl = fitControl_grid)
end_time <- Sys.time()

radialSVMTunedTime <- end_time - start_time
# Radial SVM Tuned Training Time = 20.44994 mins 

predictandCM(radial_svm_tuned,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1750   37
# Sh           38  384
# 
# Accuracy : 0.966           
# 95% CI : (0.9576, 0.9732)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.8901          
# 
# Mcnemar's Test P-Value : 1               
#                                           
#             Sensitivity : 0.9121          
#             Specificity : 0.9787          
#          Pos Pred Value : 0.9100          
#          Neg Pred Value : 0.9793          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1738          
#    Detection Prevalence : 0.1910          
#       Balanced Accuracy : 0.9454          
#                                           
#        'Positive' Class : Sh    

set.seed(100)

start_time <- Sys.time()
poly_svm <- train(label ~ ., 
                    data = trainData,
                    method = 'svmPoly',
                    metric = 'Accuracy',
                    preProc = c("center","scale"),
                    trControl = fitControl_grid)
end_time <- Sys.time()

polySVMTime <- end_time - start_time
# Poly SVM Training Time = 13.86563 mins 

predictandCM(poly_svm,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1750   53
# Sh           38  368
# 
# Accuracy : 0.9588          
# 95% CI : (0.9497, 0.9667)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.8646          
# 
# Mcnemar's Test P-Value : 0.1422          
#                                           
#             Sensitivity : 0.8741          
#             Specificity : 0.9787          
#          Pos Pred Value : 0.9064          
#          Neg Pred Value : 0.9706          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1666          
#    Detection Prevalence : 0.1838          
#       Balanced Accuracy : 0.9264          
#                                           
#        'Positive' Class : Sh  

tunegrid_svm_poly  <- expand.grid(C = c(1, 2, 3),
                                  degree = c(2,3, 4),
                                  scale = c(0.1,0.01, 0.3))

set.seed(100)

start_time <- Sys.time()
poly_svm_tuned <- train(label ~ ., 
                  data = trainData,
                  method = 'svmPoly',
                  metric = 'Accuracy',
                  preProc = c("center","scale"),
                  tuneGrid = tunegrid_svm_poly,
                  trControl = fitControl_grid)
end_time <- Sys.time()

polySVMTunedTime <- end_time - start_time
# Poly SVM Training Time = 37.21755 mins 

predictandCM(poly_svm_tuned,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1747   44
# Sh           41  377
# 
# Accuracy : 0.9615          
# 95% CI : (0.9526, 0.9692)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.8749          
# 
# Mcnemar's Test P-Value : 0.8283          
#                                           
#             Sensitivity : 0.8955          
#             Specificity : 0.9771          
#          Pos Pred Value : 0.9019          
#          Neg Pred Value : 0.9754          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1707          
#    Detection Prevalence : 0.1892          
#       Balanced Accuracy : 0.9363          
#                                           
#        'Positive' Class : Sh              
                              
###### LS-SVMs ##############################################

fitControl_lssvm <- trainControl(method = "repeatedcv",
                                number = 10,
                                repeats = 3,
                                search = 'grid')

set.seed(100)
start_time <- Sys.time()
radial_ls_svm <- train(label ~ ., 
                       data = trainData,
                       method = 'lssvmRadial',
                       metric = 'Accuracy',
                       preProc = c("center","scale"),
                       trControl = fitControl_lssvm)
end_time <- Sys.time()

radiallssvmTime <- end_time - start_time
# Radial LS-SVM training time = 3.145557 mins

predictandCM(radial_ls_svm,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1751   98
# Sh           37  323
# 
# Accuracy : 0.9389          
# 95% CI : (0.9281, 0.9485)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7903          
# 
# Mcnemar's Test P-Value : 2.418e-07       
#                                           
#             Sensitivity : 0.7672          
#             Specificity : 0.9793          
#          Pos Pred Value : 0.8972          
#          Neg Pred Value : 0.9470          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1462          
#    Detection Prevalence : 0.1630          
#       Balanced Accuracy : 0.8733          
#                                           
#        'Positive' Class : Sh   

tunegrid_ls_svm  <- expand.grid(sigma = c(0.02133332, 0.01,0.005,0.001),
                                  tau = c(0.01,0.0625, 0.005,0.001))

set.seed(100)
start_time <- Sys.time()
radial_ls_svm_tuned1 <- train(label ~ ., 
                       data = trainData,
                       method = 'lssvmRadial',
                       metric = 'Accuracy',
                       preProc = c("center","scale"),
                       tuneGrid = tunegrid_ls_svm,
                       trControl = fitControl_lssvm)
end_time <- Sys.time()

radiallssvmTunedTime1 <- end_time - start_time
# Radial LS-SVM training time = 22.03368 mins

predictandCM(radial_ls_svm_tuned1,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1752   72
# Sh           36  349
# 
# Accuracy : 0.9511          
# 95% CI : (0.9413, 0.9597)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8362          
# 
# Mcnemar's Test P-Value : 0.0007575       
#                                           
#             Sensitivity : 0.8290          
#             Specificity : 0.9799          
#          Pos Pred Value : 0.9065          
#          Neg Pred Value : 0.9605          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1580          
#    Detection Prevalence : 0.1743          
#       Balanced Accuracy : 0.9044          
#                                           
#        'Positive' Class : Sh 


###### Logistic Regression ##################################

set.seed(100)
start_time <- Sys.time()
log_reg <- train(label ~ ., 
                       data = trainData,
                       method = 'glm',
                       family = binomial(),
                       metric = 'Accuracy',
                       preProc = c("center","scale"),
                       trControl = fitControl_grid)

end_time <- Sys.time()

logregTime <- end_time - start_time
# Logistic Regression Training Time = 2.869243 secs = 0.04782 mins

predictandCM(log_reg,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh 1739   88
# Sh    49  333
# 
# Accuracy : 0.938           
# 95% CI : (0.9271, 0.9477)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7916          
# 
# Mcnemar's Test P-Value : 0.001168        
#                                           
#             Sensitivity : 0.7910          
#             Specificity : 0.9726          
#          Pos Pred Value : 0.8717          
#          Neg Pred Value : 0.9518          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1507          
#    Detection Prevalence : 0.1729          
#       Balanced Accuracy : 0.8818          
#                                           
#        'Positive' Class : Sh  

log_reg$finalModel$coefficients

######################## SMOTE SVMs ########################

set.seed(100)
start_time <- Sys.time()
smote_radial_svm <- train(label ~ ., 
                    data = smote_train,
                    method = 'svmRadial',
                    metric = 'Accuracy',
                    preProc = c("center","scale"),
                    trControl = fitControl_grid)
end_time <- Sys.time()

smote_radialSVMTime <- end_time - start_time
# Radial SVM Training Time = 4.281759 mins 

predictandCM(smote_radial_svm,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1723   25
# Sh           65  396
# 
# Accuracy : 0.9593          
# 95% CI : (0.9502, 0.9671)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8726          
# 
# Mcnemar's Test P-Value : 3.94e-05        
#                                           
#             Sensitivity : 0.9406          
#             Specificity : 0.9636          
#          Pos Pred Value : 0.8590          
#          Neg Pred Value : 0.9857          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1793          
#    Detection Prevalence : 0.2087          
#       Balanced Accuracy : 0.9521          
#                                           
#        'Positive' Class : Sh          

smote_tunegrid_svm_radial  <- expand.grid(C = c(0.5, 1, 1.5,2),
                                    sigma = c(0.01, 0.1, 0.15, 0.2))

set.seed(100)
start_time <- Sys.time()
smote_radial_svm_tuned <- train(label ~ ., 
                          data = smote_train,
                          method = 'svmRadial',
                          metric = 'Accuracy',
                          preProc = c("center","scale"),
                          tuneGrid = smote_tunegrid_svm_radial,
                          trControl = fitControl_grid)
end_time <- Sys.time()

smote_radialSVMTunedTime <- end_time - start_time
# Radial SVM Tuned Training Time = 22.17932 mins 

predictandCM(smote_radial_svm_tuned,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1733   24
# Sh           55  397
# 
# Accuracy : 0.9642          
# 95% CI : (0.9556, 0.9716)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8873          
# 
# Mcnemar's Test P-Value : 0.0007374       
#                                           
#             Sensitivity : 0.9430          
#             Specificity : 0.9692          
#          Pos Pred Value : 0.8783          
#          Neg Pred Value : 0.9863          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1797          
#    Detection Prevalence : 0.2046          
#       Balanced Accuracy : 0.9561          
#                                           
#        'Positive' Class : Sh  

set.seed(100)

start_time <- Sys.time()
smote_poly_svm <- train(label ~ ., 
                  data = smote_train,
                  method = 'svmPoly',
                  metric = 'Accuracy',
                  preProc = c("center","scale"),
                  trControl = fitControl_grid)
end_time <- Sys.time()

smote_polySVMTime <- end_time - start_time
# Poly SVM Training Time = 30.97086 mins 

predictandCM(smote_poly_svm,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1726   25
# Sh           62  396
# 
# Accuracy : 0.9606          
# 95% CI : (0.9516, 0.9683)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8765          
# 
# Mcnemar's Test P-Value : 0.0001136       
#                                           
#             Sensitivity : 0.9406          
#             Specificity : 0.9653          
#          Pos Pred Value : 0.8646          
#          Neg Pred Value : 0.9857          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1793          
#    Detection Prevalence : 0.2073          
#       Balanced Accuracy : 0.9530          
#                                           
#        'Positive' Class : Sh   

smote_tunegrid_svm_poly  <- expand.grid(C = c(0.75, 1,1.5, 2),
                                  degree = 3,
                                  scale = c(0.5,0.1,0.15,0.2))

set.seed(100)

start_time <- Sys.time()
smote_poly_svm_tuned <- train(label ~ ., 
                        data = smote_train,
                        method = 'svmPoly',
                        metric = 'Accuracy',
                        preProc = c("center","scale"),
                        tuneGrid = smote_tunegrid_svm_poly,
                        trControl = fitControl_grid)
end_time <- Sys.time()

smote_polySVMTunedTime <- end_time - start_time
# Time difference of 20.36202 mins 

predictandCM(smote_poly_svm_tuned,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1718   23
# Sh           70  398
# 
# Accuracy : 0.9579          
# 95% CI : (0.9487, 0.9659)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8691          
# 
# Mcnemar's Test P-Value : 1.842e-06       
#                                           
#             Sensitivity : 0.9454          
#             Specificity : 0.9609          
#          Pos Pred Value : 0.8504          
#          Neg Pred Value : 0.9868          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1802          
#    Detection Prevalence : 0.2119          
#       Balanced Accuracy : 0.9531          
#                                           
#        'Positive' Class : Sh               

###### SMOTE LS-SVMs ##############################################

fitControl_lssvm <- trainControl(method = "repeatedcv",
                                 number = 10,
                                 repeats = 3,
                                 search = 'grid')

set.seed(100)
start_time <- Sys.time()
smote_radial_ls_svm <- train(label ~ ., 
                       data = smote_train,
                       method = 'lssvmRadial',
                       metric = 'Accuracy',
                       preProc = c("center","scale"),
                       trControl = fitControl_lssvm)
end_time <- Sys.time()

smote_radiallssvmTime <- end_time - start_time
# Radial LS-SVM training time = 4.434833 mins

predictandCM(smote_radial_ls_svm,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1660   20
# Sh          128  401
# 
# Accuracy : 0.933           
# 95% CI : (0.9218, 0.9431)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8022          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.9525          
#             Specificity : 0.9284          
#          Pos Pred Value : 0.7580          
#          Neg Pred Value : 0.9881          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1815          
#    Detection Prevalence : 0.2395          
#       Balanced Accuracy : 0.9405          
#                                           
#        'Positive' Class : Sh 

smote_tunegrid_ls_svm  <- expand.grid(sigma = c(0.02133332, 0.01,0.005,0.001),
                                tau = c(0.01,0.0625, 0.005,0.001))

set.seed(100)
start_time <- Sys.time()
smote_radial_ls_svm_tuned <- train(label ~ ., 
                              data = smote_train,
                              method = 'lssvmRadial',
                              metric = 'Accuracy',
                              preProc = c("center","scale"),
                              tuneGrid = smote_tunegrid_ls_svm,
                              trControl = fitControl_lssvm)
end_time <- Sys.time()

smote_radiallssvmTunedTime <- end_time - start_time
# Radial LS-SVM training time = 11.17927 mins

predictandCM(smote_radial_ls_svm_tuned,testDatax)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  NSh   Sh
# NSh        1630   18
# Sh          158  403
# 
# Accuracy : 0.9203          
# 95% CI : (0.9082, 0.9313)
# No Information Rate : 0.8094          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7709          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.9572          
#             Specificity : 0.9116          
#          Pos Pred Value : 0.7184          
#          Neg Pred Value : 0.9891          
#              Prevalence : 0.1906          
#          Detection Rate : 0.1824          
#    Detection Prevalence : 0.2540          
#       Balanced Accuracy : 0.9344          
#                                           
#        'Positive' Class : Sh              
                                
######################## ROC Curves #########################
par(pty = "s")

# Random Forest
pred <- predict(random_forest, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# Adaboost
pred <- predict(adaboost, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# Adaboost.M1
pred <- predict(adaboost.m1, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# XGBoost
pred <- predict(xgb, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# Adabag
pred <- predict(adabag, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# SMOTE Random Forest
pred <- predict(smote_rf, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# SMOTE Adaboost
pred <- predict(smote_adaboost, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# SMOTE Adaboost.M1
pred <- predict(smote_adaboost.m1, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# SMOTE XGBoost
pred <- predict(smote_xgb, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# SMOTE Adabag
pred <- predict(smote_adabag, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# XGBoost Tune
pred <- predict(xgb_tune, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# SMOTE XGBoost Tune
pred <- predict(smotexgb_tune, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# Linear SVM
pred <- predict(linear_svm, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# Radial SVM 
pred <- predict(radial_svm_tuned, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# Poly SVM
pred <- predict(poly_svm_tuned, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# Radial LS-SVM
pred <- predict(radial_ls_svm, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)
## Doesn't Work 

# Logistic Regression
pred <- predict(log_reg, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# SMOTE Radial SVM
pred <- predict(smote_radial_svm_tuned, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# SMOTE Poly SVM
pred <- predict(smote_poly_svm, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

# SMOTE LS-SVM
pred <- predict(smote_radial_ls_svm, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)
#Doesn't Work 

#Combined ROC Curves 
# Random Forest, Adaboost, XGBoost, Adabag, Radial SVM
pred <- predict(random_forest, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

pred <- predict(adaboost, testDatax, type = "prob")
plot.roc(testDatay, pred[,2], col="#F71313", add = TRUE,
    lwd=4, print.auc=TRUE, print.auc.y = 0.45)

pred <- predict(xgb_tune, testDatax, type = "prob")
plot.roc(testDatay, pred[,2], col="#0EF763", add=TRUE,
    lwd=4, print.auc=TRUE, print.auc.y=0.4)

pred <- predict(adabag, testDatax, type = "prob")
plot.roc(testDatay, pred[,2], col="#F40EFC", add=TRUE,
    lwd=4, print.auc=TRUE, print.auc.y=0.35)

# Radial SVM 
pred <- predict(radial_svm_tuned, testDatax, type = "prob")
plot.roc(testDatay, pred[,2], col="#F6CD00", add=TRUE,
         lwd=4, print.auc=TRUE, print.auc.y=0.3)

# Legend 
plot(NULL ,xaxt='n',yaxt='n',bty='n',ylab='',xlab='', xlim=0:1, ylim=0:1)
legend("topleft", legend =c("Random Forest", "AdaBoost", "XGBoost", "AdaBag", "Radial SVM"), pch=16, pt.cex=3, cex=0.6, bty='n',
       col=c("#377eb8", "#F71313","#0EF763","#F40EFC","#F6CD00"))


# SMOTE 
pred <- predict(smote_rf, testDatax, type = "prob")
roc(testDatay, pred[,2], plot = TRUE, legacy.axes=TRUE, col="#377eb8",
    lwd=4, print.auc=TRUE)

pred <- predict(smote_adaboost.m1, testDatax, type = "prob")
plot.roc(testDatay, pred[,2], col="#F71313", add = TRUE,
         lwd=4, print.auc=TRUE, print.auc.y = 0.45)

pred <- predict(smotexgb_tune, testDatax, type = "prob")
plot.roc(testDatay, pred[,2], col="#0EF763", add=TRUE,
         lwd=4, print.auc=TRUE, print.auc.y=0.4)

pred <- predict(smote_adabag, testDatax, type = "prob")
plot.roc(testDatay, pred[,2], col="#F40EFC", add=TRUE,
         lwd=4, print.auc=TRUE, print.auc.y=0.35)

# Radial SVM 
pred <- predict(smote_radial_svm_tuned, testDatax, type = "prob")
plot.roc(testDatay, pred[,2], col="#F6CD00", add=TRUE,
         lwd=4, print.auc=TRUE, print.auc.y=0.3)

# SMOTE Poly SVM
pred <- predict(smote_poly_svm, testDatax, type = "prob")
plot.roc(testDatay, pred[,2], col="#04A809", add=TRUE,
         lwd=4, print.auc=TRUE, print.auc.y=0.25)

# Legend 
plot(NULL ,xaxt='n',yaxt='n',bty='n',ylab='',xlab='', xlim=0:1, ylim=0:1)
legend("topleft", legend =c("Random Forest", "AdaBoost", "XGBoost", "AdaBag", "Radial SVM", "Poly SVM"), pch=16, pt.cex=1.8, cex=0.6, bty='n',
       col=c("#377eb8", "#F71313","#0EF763","#F40EFC","#F6CD00", "#04A809" ))

par(pty="m")

#############################################################
#Feature Importance plot 
# Adaboost 
adaboost.importance <- data.frame(Features = c("abin", "cm", "count1", "count2", "count3", "cvbin", "exp", "frqbin", "kurt", "mav", "tci", "vfleak"),
                                  Importance = c(0.04910409,0.02154762,0.05494384,0.06555726,0.13907403,0.02721062,0.02607797,0.06483063,0.04313501,0.17306942,0.0895939,0.24585558))
  
ggplot(data =adaboost.importance, aes(x=reorder(Features,Importance), y = Importance, width = 0.5)) +
       geom_bar(stat ="identity", aes(fill = Features)) +
       scale_fill_manual(values=c("vfleak" = '#0D96D1', "mav" = '#2EC423', "count3" = "#2EC423", "tci" = "#2EC423", "count2" = "#F52A1B", "frqbin" = "#F52A1B", "count1" = "#F52A1B", "abin" = "#F52A1B", "kurt" = "#F52A1B","cvbin"= "#F52A1B","exp" = "#F52A1B","cm" = "#F52A1B")) +
       labs(x = "Features") + theme(legend.position = "none") +
       theme(plot.title = ggplot2::element_text(lineheight=.9, face="bold"), panel.grid.major.y = ggplot2::element_blank()) +
       coord_flip() + ggtitle("AdaBoost") + ylim(0.0,0.35)


# SMOTE RF
smote.rf.importance <- data.frame(Features = c("abin", "cm", "count1", "count2", "count3", "cvbin", "exp", "frqbin", "kurt", "mav", "tci", "vfleak"),
                                  Importance = c(0.04765381,0.026386386,0.112277765,0.027894568,0.168205832,0.062380059,0.026746276,0.057548292,0.090867988,0.074807515,0.073033963,0.232197546))

ggplot(data =smote.rf.importance, aes(x=reorder(Features,Importance), y = Importance, width = 0.5)) +
  geom_bar(stat ="identity", aes(fill = Features)) +
  scale_fill_manual(values=c("vfleak" = '#0D96D1', "mav" = '#F52A1B', "count3" = "#0D96D1", "tci" = "#F52A1B", "count2" = "#F52A1B", "frqbin" = "#F52A1B", "count1" = "#2EC423", "abin" = "#F52A1B", "kurt" = "#F52A1B","cvbin"= "#F52A1B","exp" = "#F52A1B","cm" = "#F52A1B")) +
  labs(x = "Features") + theme(legend.position = "none") +
  theme(plot.title = ggplot2::element_text(lineheight=.9, face="bold"), panel.grid.major.y = ggplot2::element_blank()) +
  coord_flip() + ggtitle("SMOTE Random Forest") + ylim(0.0,0.35)

# SMOTE AdaBoost 
smote.adaboost.importance <- data.frame(Features = c("abin", "cm", "count1", "count2", "count3", "cvbin", "exp", "frqbin", "kurt", "mav", "tci", "vfleak"),
                                  Importance = c(0.0433621,0.02659956,0.04940336,0.03692,0.096956681,0.03934783,0.065904781,0.079370171,0.040998,0.142292401,0.083514631,0.295330483))

ggplot(data =smote.adaboost.importance, aes(x=reorder(Features,Importance), y = Importance, width = 0.5)) +
  geom_bar(stat ="identity", aes(fill = Features)) +
  scale_fill_manual(values=c("vfleak" = '#0D96D1', "mav" = '#0D96D1', "count3" = "#2EC423", "tci" = "#F52A1B", "count2" = "#F52A1B", "frqbin" = "#F52A1B", "count1" = "#F52A1B", "abin" = "#F52A1B", "kurt" = "#F52A1B","cvbin"= "#F52A1B","exp" = "#F52A1B","cm" = "#F52A1B")) +
  labs(x = "Features") + theme(legend.position = "none") +
  theme(plot.title = ggplot2::element_text(lineheight=.9, face="bold"), panel.grid.major.y = ggplot2::element_blank()) +
  coord_flip() + ggtitle("SMOTE AdaBoost") + ylim(0.0,0.35)


#############################################################
stopCluster(cl)
