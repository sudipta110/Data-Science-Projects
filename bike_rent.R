#removing all the stored objects
rm(list=ls())

#set current working directory
setwd('D:/edwisor/Project Docs')

#get working directory
getwd()

#Installing required packages at a time
install.packages(c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", 
                   "Information","MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees'))

install.packages(c("dplyr","plyr","reshape","ggplot2","data.table","GGally"))

# Install  Require libraries
library("dplyr")
library("plyr")
library("ggplot2")
library("data.table")
library("GGally")
library(corrgram)

#load data in R - reading .csv file
bike_data=read.csv('day.csv',header=TRUE)

#first five rows of data
head(bike_data)

#'cnt' is target variable and other variables are independent  variable(or predictors)

#summary of data
summary(bike_data)

#structure of  data
str(bike_data)

#****************Univariate Analysis***************#

# function for univariate distribution
univ_dist <- function(num_var) {
  
  
  ggplot(data=bike_data)+
    geom_histogram(aes(x=num_var,y=..density..),
                   col="red",
                   fill="blue",
                   alpha=0.3)+
      geom_density(aes(x=num_var,y=..density..))
  
}


# distribution of  target variable 'cnt'
univ_dist(bike_data$cnt)

# distrubution of  independent variable 'temp'
univ_dist(bike_data$temp)

# distrubution of  independent variable 'atemp'
univ_dist(bike_data$atemp)

# distrubution of  independent variable 'hum'
univ_dist(bike_data$hum)

# distrubution of  independent variable 'windspeed'
univ_dist(bike_data$windspeed)

# distrubution of  independent variable 'casual'
univ_dist(bike_data$casual)

# distrubution of  independent variable 'casual'
univ_dist(bike_data$registered)


# Visualize categorical Variable 'holiday' 

ggplot(bike_data) +
  geom_bar(aes(x=holiday),fill="green",col="red",alpha=0.3)

# the visualization is showing that almost all the  bike rentals are happening  on holidays

# Visualize categorical Variable 'weekday' 

ggplot(bike_data) +
  geom_bar(aes(x=weekday),fill="green",col="red",alpha=0.3) 

# the visualization is showing bike rental counts are same on all weekdays

# Visualize categorical Variable 'weathersit' 

ggplot(bike_data) +
  geom_bar(aes(x=weathersit),fill="green",col="red",alpha=0.3) 

# bike rental count  is more when  weather is " Clear, Few clouds, Partly cloudy, Partly cloudy"

#*****************Multivariate Analysis*********************#

ggpairs(bike_data[,c('atemp','temp','hum','windspeed','casual','registered','cnt')],title = 'Multivariate Analysis')

#*******************Missing Value Analysis***********************#

#create dataframe with missing percentage
missing_val = data.frame(apply(bike_data,2,function(var){sum(is.na(var))}))

#converting rownames into columns
missing_val$columns=row.names(missing_val)
row.names(missing_val)=NULL

#renaming the varibale name
names(missing_val)[1] =  "Missing_percentage"

#calculate percentage
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(bike_data)) * 100

#So , no missing  values are presnt in the data set

#*****************Outlier Analysis*******************#

# detect outliers in  'casual' , 'registered' variables

# boxplot for  casual  variable

ggplot(data = bike_data, aes(x = "cnt", y = casual)) + 
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "blue",alpha=0.3,outlier.shape=18,
               outlier.size=2, notch=FALSE) +
  theme(legend.position="bottom")+
  ggtitle(paste("Box plot for casual"))

# Boxplot is showing there are few outliers in  casual variables

# boxplot for  Registered  variable

ggplot(data = bike_data, aes(x = "", y = registered)) + 
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
               outlier.size=2, notch=FALSE) +
  theme(legend.position="bottom")+
  ggtitle(paste("Box plot for registered"))

# there  is no outliers  in registered variables


#***************Treat Outliers*******************#

# relationship between causal and cnt variables before  outlier treatment

ggplot(bike_data, aes(x= casual,y=cnt)) +
  geom_point()+
  geom_smooth()

cor(bike_data$casual,bike_data$cnt) 

# #Remove outliers using boxplot method

val_out = bike_data$casual[bike_data$casual %in% boxplot.stats(bike_data$casual)$out]
bike_data = bike_data[which(!bike_data$casual %in% val_out),]

# Boxplot after removing  outliers

ggplot(data = bike_data, aes(x = "cnt", y = casual)) + 
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "blue" ,alpha=0.3,outlier.shape=18,
               outlier.size=2, notch=FALSE) +
  theme(legend.position="bottom")+
  ggtitle(paste("Box plot for casual"))

# verify the relationship after  outliers
ggplot(bike_data, aes(x= casual,y=cnt)) +
  geom_point()+
  geom_smooth()

cor(bike_data$casual,bike_data$cnt) 

#********************Feature Selection or dimension reduction*******************#


# verify correleation between Numeric variable

corrgram(bike_data[,c('temp','atemp','hum','windspeed','casual','registered','cnt')], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


# correlation matrix  stating  'temp' and 'atemp' having strong relationship
# and there is no  relationship between 'hum' and 'cnt'

#  dimensional  reduction

bike_data = subset(bike_data,select=-c(atemp,hum))

########################  Normality  check #################################

#Normalisation

cnames = c("casual","registered")

for(i in cnames){
  print(i)
  bike_data[,i] = (bike_data[,i] - min(bike_data[,i]))/
    (max(bike_data[,i] - min(bike_data[,i])))
}

#check value after normalization
bike_data$casual
bike_data$registered


#*******************Model Development***********************#

#Divide data into train and test using stratified sampling method
set.seed(1234)
library(caret)
train.index = createDataPartition(bike_data$cnt, p = .80, list = FALSE)
train = bike_data[ train.index,]
test  = bike_data[-train.index,]

train_bike = train[,c("season" ,"yr" ,"mnth","holiday","weekday","workingday","weathersit","temp","windspeed","casual","registered","cnt")]

test_bike = test[,c("season" ,"yr" ,"mnth","holiday","weekday","workingday","weathersit","temp","windspeed","casual","registered","cnt")]

#******************develop Decision tree model*****************#

#rpart for regression
DT = rpart(cnt ~ ., data = train_bike, method = "anova")

#Predict for new test cases
predictions_DT = predict(DT, test_bike[,-12])

print(DT)

#  plotting decision tree

par(cex= 1)
plot(DT)
text(DT)

#***************Evaluate  Decision tree****************#

#MAPE
#Evalute model using MAPE

MAPE(test_bike[,12], predictions_DT)

#Error Rate: 10.37541
#Accuracy: 89.62459

#Evaluate  Model using RMSE

RMSE(test_bike[,12], predictions_DT)

#RMSE = 453.6728

#*****************Random Forest**********************#

RF_1=randomForest(cnt ~ . , data = train_bike)

RF_1

plot(RF_1)
#**********************Evaluate Random Forest******************#

#Predict for new test cases
predictions_RF_1 = predict(RF_1, test_bike[,-12])

MAPE(test_bike[,12], predictions_RF_1)

#Error Rate: 4.537826
#Accuracy: 95.462174

RMSE(test_bike[,12], predictions_RF_1)

#RMSE = 205.3907

#*****************Parameter Tuning for random forest****************#

RF_2=randomForest(cnt ~ . , data = train_bike,mtry =7,ntree=500 ,nodesize =10 ,importance =TRUE)

RF_2

plot(RF_2)

#Predict for new test cases
predictions_RF_2 = predict(RF_2, test_bike[,-12])

MAPE(test_bike[,12], predictions_RF_2)

#Error Rate: 2.076463
#Accuracy: 97.923537

RMSE(test_bike[,12], predictions_RF_2)

#RMSE = 118.7641

# check Variable  Importance 

RFimp = importance(RF_2)

RFimp
# sort variable  

sort_var = names(sort(RFimp[,1],decreasing =T))

sort_var
# draw varimp plot 

varImpPlot(RF_2,type = 2)

#*****************Tuning Random Forest Dimensional reduction********************#

#   remove four variables  which is  contributing  less

#"season"     "weathersit" "windspeed"  "holiday"   are removing and  developing the  new model

train_bike_RF = train[,c("yr" ,"mnth","weekday","workingday","temp","casual","registered","cnt")]
test_bike_RF = test[,c("yr" ,"mnth","weekday","workingday","temp","casual","registered","cnt")]

# Develop Random Forest  Model

RF_3=randomForest(cnt ~ . , data = train_bike_RF,mtry =7,ntree=500 ,nodesize =10 ,importance =TRUE)

RF_3

plot(RF_3)

#Predict for new test cases
predictions_RF_3 = predict(RF_3, test_bike_RF[,-8])

MAPE(test_bike_RF[,8], predictions_RF_3)

#Error Rate: 1.621891
#Accuracy: 98.378109

RMSE(test_bike_RF[,8], predictions_RF_3)

#RMSE = 88.53711

#****************Develop  Linear Regression Model*******************#

#check multicollearity
install.packages('usdm')
library(usdm)

vifcor(train_bike[,-12], th = 0.9)
# Correleation between two variables is 'season' and 'mnth' is 0.82 so, removing 'season' variable from the model

train_bike_linear = train[,c("yr" ,"mnth","holiday","weekday","workingday","weathersit","temp",
                             "windspeed","casual","registered","cnt")]
test_bike_linear = test[,c("yr" ,"mnth","holiday","weekday","workingday","weathersit","temp",
                           "windspeed","casual","registered","cnt")]

# develop Linear Regression  model

#run regression model
lm_model = lm(cnt ~., data = train_bike_linear)

#Summary of the model
summary(lm_model)

# observe the  residuals and   coefficients  of the linear regression model

# Predict  the Test data 

#Predict
predictions_LR = predict(lm_model, test_bike_linear[,-11])

predictions_LR
# Evaluate Linear Regression Model

MAPE=function(target,predictions){
  mean(abs((target-predictions)/target))*100
}

MAPE(test_bike_linear[,11], predictions_LR)

#Error Rate: 1.065695e-13
#Accuracy: 99.9 + accuracy

RMSE(test_bike_linear[,11], predictions_LR)

#RMSE = 3.351778e-12

# COnclusion  For this Dataset  Linear Regression Accuracy  is '99.9'
# and RMSE = 3.351778e-12 
