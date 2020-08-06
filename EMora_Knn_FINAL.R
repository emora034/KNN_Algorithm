
rm(list = ls())
library(dplyr)
library(skimr)
library(keep)
library(tidyr)
library(tictoc)
#We use the set.seed() to produce reproducible results
set.seed(100392505)

#Heart Disease Data Set
#This dataset contains 14 variables, where data was collected from 303 patients who were 
#admitted to a hospital. The "goal/target" field refers to the presence of heart disease 
#in the patient (1=yes; 0=no). The variables' information is as follows:
#1. age: The person's age in years
#2. sex: The person's sex (1 = male, 0 = female)
#3. cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
#4. trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
#5. chol: The person's cholesterol measurement in mg/dl
#6. fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false) 
#7. restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
#8. thalach: The person's maximum heart rate achieved during Stress Test (exercise)
#9. exang: Exercise induced angina (1 = yes; 0 = no)
#10. oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot)
#11. slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
#12. ca: The number of major vessels (0-3) colored by flourosopy 
#13. thal: A blood disorder called thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)
#14. goal/target: Heart disease (0 = no, 1 = yes)

Heart<-read.csv("Heart Disease Data.csv", header = TRUE)

# Let's check for NA values
colSums(is.na(Heart))
#deleting thal variable as there's missing information on the labels
#source claims 3 factors, but there are 4. 
Heart<-Heart[,-13]

#Our target variable is categorical and we need to define its factors
Heart$target=factor(Heart$target)
levels(Heart$target)=c("Healthy","Disease")

########################################
########################################
#----------------Split Data-------------
#Before splitting the dataset we scale all variables except our target variable
heart<-data.frame(scale(Heart[,c(1,2,3,4,5,6,7,8,9,10,11,12)]),Heart[,13])
#The data will be split in 50%, 25% and 25% for a training, testing, and validation set
#respectively
intrain <- sample(3, nrow(heart), replace = TRUE, prob = c(0.5, 0.25,0.25))

train <-heart[intrain==1,]
test <- heart[intrain==2,]
validate<-heart[intrain==3,]

###################################################################################
############################ My KNN Function ######################################

#The function has 5 inputs:
# 1. train: data set used for training, in our case 50% of the original dataset, without the
#     target variable (1x12 matrix of attributes)
# 2. target: the target is a column vector extracted from the training dataset, it includes
#     the classification (Disease or Healthy) only.
# 3. observation: this input takes a matrix of attributes only (testing or validation set),
#     so that our function can predict the target values.
# 4. k_val: this input takes an integer such that we check the kth closest neighbors 
#     in order to determine the category
# 5. dmethod: takes a character input where we specify how to calculate the distances,
#     in our case there are only two options "euclidean" or "manhattan".

my_KNN<-function(train,target,observation, k_val, dmethod="euclidean"){
  #first we establish the calculation of the distances given the chosen input method
  if(dmethod == "euclidean"){
    distancias = sqrt(rowSums(sweep(train,2,as.numeric(observation))^2))
  }
  if(dmethod == 'manhattan'){
    distancias = rowSums(abs(sweep(train,2,as.numeric(observation))))
  }
  #the distances are ordered in ascending order so that we can establish the category to
  #be predicted, given its closeness to the majority of the neighbors.
  indices = order(distancias)
  target_ordered = target[indices]
  k_obs = target_ordered[1:k_val]
  prediction = names(which.max(table(k_obs)))
  return(prediction)
}

#Funtion accuracy - we include the computation of the accuracy of the predictions. Here
#the prediction is held against the true reported values, if it's correctly labeled then
#we add it to the count of "correct" labels.
accuracy<-function(test){
  correct=0
  for(i in c(1:nrow(test))){
    if(test[i,1]==test[i,2]){
      correct=correct+1
    }
  }
  real_accuracy=correct/nrow(test)*100
  return(real_accuracy)
}

#--------Let's do a quick implementation of the function to check if it works--------#
################# K = 1 & euclidian methods
tries = c()
for(i in 1:nrow(test)){
  tries = c(tries,my_KNN(train[,-13],train[,13],test[i,-13],1,dmethod="euclidean"))}

check = data.frame(as.factor(tries),test[,13])
accuracy(check)

################# K = 1 & manhattan method
tries = c()
for(i in 1:nrow(test)){
  tries = c(tries,my_KNN(train[,-13],train[,13],test[i,-13],1,dmethod="manhattan"))}

check = data.frame(as.factor(tries),test[,13])
accuracy(check)

################# K = 2 & euclidean method
tries = c()
for(i in 1:nrow(test)){
  tries = c(tries,my_KNN(train[,-13],train[,13],test[i,-13],2,dmethod="euclidean"))}

check = data.frame(as.factor(tries),test[,13])
accuracy(check)

################# K = 3 & euclidean method
tries = c()
for(i in 1:nrow(test)){
  tries = c(tries,my_KNN(train[,-13],train[,13],test[i,-13],3,dmethod="euclidean"))}

check = data.frame(as.factor(tries),test[,13])
accuracy(check)

################# K = 4 & manhattan method
tries = c()
for(i in 1:nrow(test)){
  tries = c(tries,my_KNN(train[,-13],train[,13],test[i,-13],4,dmethod="manhattan"))}

check = data.frame(as.factor(tries),test[,13])
accuracy(check)

################# K = 5 & euclidean method
tries = c()
for(i in 1:nrow(test)){
  tries = c(tries,my_KNN(train[,-13],train[,13],test[i,-13],5,dmethod="euclidean"))}

check = data.frame(as.factor(tries),test[,13])
accuracy(check)

#------------------------------------------------------------------------------------------#
############################# Validation Set + 100 Samples#################################
#In this section we will now use the validation set to find the value of K which yields
#the highest accuracy. For this we set up a loop where we test values of K from 1-10, while
#resampling/partitioning the data, at random, 100 times. 

#we first try the loop only taking values of K from 1-10, with the exising sample split
#the results are stored into a datafram "data_acc". The dataframe successfuly stored the
#values of K and its respective accuracy.
data_acc = data.frame()
for(k in 1:10){
  tries = c()
  for(i in 1:nrow(validate)){
    tries = c(tries,my_KNN(train[,-13],train[,13],validate[i,-13],k,dmethod="euclidean"))}
  
  check = data.frame(as.factor(tries),validate[,13])
  data_acc = rbind(data_acc,c(k,accuracy(check)))
}
colnames(data_acc) = c('k','Accuracy')

#---------------------------------------------------------------------------------------#
#Now we add the random sampling (repeated 100 times) to the above loop

#initialize the dataframe where all accuracies will be recorded along with their K-val

data_100 = data.frame(1:10)
colnames(data_100) = 'k'
#initialize loop with 1-100 resamplings
tic()
for(j in 1:100){
  intrain <- sample(3, nrow(heart), replace = TRUE, prob = c(0.5, 0.25,0.25))
  
  train <-heart[intrain==1,]
  test <- heart[intrain==2,]
  validate<-heart[intrain==3,]
  #initialize empty col vector where accuracies will be stored
  accuracies = c()
  #include loop for k values to be implemented with the validation set
  for(k in 1:10){
    tries = c()
    for(i in 1:nrow(validate)){
      tries = c(tries,my_KNN(train[,-13],train[,13],validate[i,-13],k,dmethod="euclidean"))}
    #include accuracy computation by sample per K-value
    check = data.frame(as.factor(tries),validate[,13])
    accuracies = c(accuracies,accuracy(check))
  }
  #this dataframe will have dimension 10x101 - 10 k-vals and 100 accuracies by k-val.
  data_100 = cbind(data_100,accuracies)
  
}
toc()
#we create a new dataframe were accuracies are grouped by k value and their means
#are computed
final_accuracies = data.frame(data_100$k,apply(data_100[,2:100],1,mean))
colnames(final_accuracies) = c('k','mean accuracy')
#sort(final_accuracies[,2])
final_accuracies[order(final_accuracies$`mean accuracy`, decreasing = TRUE),]
View(final_accuracies)

library("foreach")
library("doParallel")

cl=makeCluster(8)
registerDoParallel(cl)

tic()
parallel = foreach(j=1:100, .combine = 'cbind')%dopar%{
  intrain <- sample(3, nrow(heart), replace = TRUE, prob = c(0.5, 0.25,0.25))
  
  train <-heart[intrain==1,]
  test <- heart[intrain==2,]
  validate<-heart[intrain==3,]
  accuracies = c()
  for(k in 1:10){
    tries = c()
    for(i in 1:nrow(validate)){
      tries = c(tries,my_KNN(train[,-13],train[,13],validate[i,-13],k,dmethod="euclidean"))}
    
    check = data.frame(as.factor(tries),validate[,13])
    accuracies = c(accuracies,accuracy(check))
  }
  accuracies
}
toc()
stopCluster(cl)

parallel = cbind(1:10,parallel)
final_accuracies_parallel = data.frame(parallel[,1],apply(parallel[,2:100],1,mean))
colnames(final_accuracies_parallel) = c('k','mean accuracy')
sort(final_accuracies_parallel[,2])



#####################################################################################
#--------Attempt to include graph output given input=TRUE--------#
#I create a PCA where we plot PCA1 vs. PCA2 and then we add
#one point into the graph such that it should be close to neighbors of the
#same class (disease or healthy)

library(ggfortify)
pca_res <- prcomp(train[,1:12])

autoplot(pca_res, data = train, colour = 'Heart...13.')+
  geom_point(aes(x=pca_res$x[nrow(pca_res$x),1], y=pca_res$x[nrow(pca_res$x),1]), colour="blue")

View(pca_res$x)

my_KNN<-function(train,target,observation, k_val, dmethod="euclidean",plot = FALSE){
  if(dmethod == "euclidean"){
    distancias = sqrt(rowSums(sweep(train,2,as.numeric(observation))^2))
  }
  if(dmethod == 'manhattan'){
    distancias = rowSums(sweep(train,2,abs(as.numeric(observation))))
  }
  indices = order(distancias)
  target_ordered = target[indices]
  k_obs = target_ordered[1:k_val]
  prediction = names(which.max(table(k_obs)))
  if(plot == TRUE){
    train2 = rbind(train,observation)
    pca_res <- prcomp(train2)
    targets = rbind(target,prediction)
    dims = data.frame(pca_res$x[,1],pca_res$x[,2],targets)
    colnames(dims) = c('x','y','Condition')
    plot = ggplot(dims,aes(x=x,y=y,color=Condition))+geom_point()+geom_point(aes(x = pca_res$x[nrow(pca_res$x),1], y = pca_res$x[nrow(pca_res$x),2], col = 'Observation'))
    return(list(plot, prediction))
  }
  return(prediction)
}

#it would only take one point 
my_KNN(train[,-13],train[,13],test[1,-13],4,dmethod="manhattan",plot = TRUE)


train2 = rbind(train,test[1,])
pca_res <- prcomp(train2[,1:12])
View(pca_res$x)
pca_res$x[nrow(pca_res$x),1]
