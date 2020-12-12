# Q:Predicting the burned area of Forest Fires using Neural Network

# Load the forest fire data
forestfires<-read.csv("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\Neural Network-R\\forestfires.csv")
FF<-forestfires
View(FF)
str(FF)

# Removing the columns 1 & 2 as they seems insignificant in model 
FF <- FF[,c(-1,-2)]
summary(FF)

# Convert the size into categorical value and further to integer data type
 FF$size_category <- as.factor(FF$size_category)
 table(FF$size_category)
 FF$size_category = as.integer(ifelse(FF$size_category == "small", "0", "1"))
 View(FF)
 str(FF)
 
 # custom normalization function
 normalize <- function(x) { 
   return((x - min(x)) / (max(x) - min(x)))
 }
 
# apply normalization to entire data frame
forest_norm <- as.data.frame(lapply(FF, normalize))
View(forest_norm)
str(forest_norm)

# Installation of required packages
install.packages(c("nnet","neuralnet", "caret", "ggplot2"))
install.packages("corrplot")
install.packages("ggvis")
install.packages("plyr")
install.packages("psych")
library(nnet)
library(neuralnet)
library(caret)
library(ggplot2)
library(plyr)
library(psych)
library(ggvis)
library(corrplot)

# Visualization of data to understand behaviours of different variables
attach(forest_norm)
forest_norm %>% ggvis(~FFMC, ~DMC,fill = ~forest_norm$size_category) %>% layer_points()
forest_norm %>% ggvis(~FFMC, ~DC,fill = ~ forest_norm$size_category) %>% layer_points()
# FFMC is mostly at higher side irrespective of different values of DMC and DC
forest_norm %>% ggvis(~temp, ~RH,fill = ~forest_norm$size_category) %>% layer_points()
# As the temperature increases RH decreases
forest_norm %>% ggvis(~RH, ~rain,fill = ~forest_norm$size_category) %>% layer_points()
# The RH does not get affected when there is no rain at the situation of any forest fire
forest_norm %>% ggvis(~area, ~RH,fill = ~forest_norm$size_category) %>% layer_points()
# At low or moderate values of RH, there is more forest fires
forest_norm %>% ggvis(~area, ~temp,fill = ~forest_norm$size_category) %>% layer_points()
# At high temperature, forest gets fired
forest_norm%>% ggvis(~area, ~rain,fill = ~forest_norm$size_category) %>% layer_points()
# In no rain condition the forest gets fired 
forest_norm %>% ggvis(~area, ~wind,fill = ~forest_norm$size_category) %>% layer_points()
# Mostly in medium wind, there is a fire in the forest 
ggplot(forest_norm) + geom_histogram(aes(FFMC), binwidth = 0.1) + ggtitle("FFMC") 
ggplot(forest_norm) + geom_histogram(aes(DMC), binwidth = 0.1) + ggtitle("DMC") 
ggplot(forest_norm) + geom_histogram(aes(DC), binwidth = 0.1) + ggtitle("DC")
# FFMC, DMC and DC do not follow normal distribution.
# However, FFMC remains always in higher side
ggplot(forest_norm) + geom_histogram(aes(temp), binwidth = 0.1) + ggtitle("Temperature") 
# Temperature follows normal distribution due to variation in temperature throughout the year
ggplot(forest_norm) + geom_histogram(aes(RH), binwidth = 0.1) + ggtitle("RH") 
# Relative Humidity distribution is right hand skewed;
# It indicates that it remains at higher side
ggplot(forest_norm) + geom_histogram(aes(rain), binwidth = 0.1) + ggtitle("Rain") 
# There is very little rain as indicated from histogram
ggplot(forest_norm) + geom_histogram(aes(wind), binwidth = 0.1) + ggtitle("Wind")
# Wind blow distribution is at higher side
hist(forest_norm$area)
# There are lot of zeros. Therefore, we need to check the data by log transformation
forest_norm1<-mutate(forest_norm,y=log(area+1))
hist(forest_norm1$y)
# Let's find out correlation among the variables
corrplot(cor(forest_norm))
# Individual variables have wide difference in their data range.
# There are positive relations among FFMC,DMC,DC ISI and Temp.
# There is a strong negative correlation between temp and RH

# Splitting dataset into training and testing
set.seed(7)
inlocalTraining<-createDataPartition(forest_norm$size_category,p=0.70,list=F)
training <- forest_norm[inlocalTraining,]
testing <- forest_norm[-inlocalTraining,]
View(training)
View(testing)

## Training a model on the data ----

# simple ANN with only a single hidden neuron
colnames(training)
FF_model1 <- neuralnet(size_category ~., data = training)

# visualize the network topology
windows();plot(FF_model1)

## Evaluating model performance ----
# obtain model results
model_results <- compute(FF_model1,testing[1:28])
predicted_area <- model_results$net.result
cor(predicted_area, testing$size_category)
# Correlation between predicted and test area is 0.94 %
# Accuracy of the model is 0.94

## To check accuracy improvement by Increasing complexity of network 
# with 6 hidden neurons 
FF_model2 <- neuralnet(size_category ~., data = training, hidden =c(4,2))
# plot the network to visualise topology
windows();plot(FF_model2)
# evaluate the results
model_results2 <- compute(FF_model2, testing[1:28])
predicted_area2 <- model_results2$net.result
cor(predicted_area2, testing$size_category)
# Correlation between predicted area and test area decreases slightly
# Accuracy slightly reduces to 0.90

# Further Improvement in the model by removing the insignificant variables
colnames(training)
# Building model by considering first 9 important variables like
# "FFMC", "DMC","DC","ISI","temp", "RH", "Wind", "rain", "area"
# Create training and test data after removal of insignificant variables
training_final <- training[,-c(10:28)]
testing_final <- testing[,-c(10:28)]
View(training_final)
View(testing_final)
# Building ANN model with significant variables only
FF_model3 <- neuralnet(size_category ~., data = training_final)
# plot the network
windows();plot(FF_model3)
# evaluate the results
model_results3 <- compute(FF_model3, testing_final[1:9])
predicted_area3 <- model_results3$net.result
cor(predicted_area3, testing_final$size_category)
# correlation between the predicted area and testing is 0.95 
# Accuracy of the model is 0.95

## To chcek improvement of model by Increasing complexity of network 
# with 6 hidden neurons
FF_model4 <- neuralnet(size_category ~., data = training_final, hidden =c(4,2))
# plot the network
windows();plot(FF_model4)
# evaluate the results
model_results4 <- compute(FF_model4,testing_final[1:9])
predicted_area4 <- model_results4$net.result
cor(predicted_area4, testing_final$size_category)
# correlation between the predicted area and size category has increased to 0.99
# Accuracy of the model is 0.99

# CONCLUSION :
# Neural Network Model 4 is best model with two hidden layers of 4 and 2 neurons
# The accuracy is 0.99 by using ANN technique.
# SVM algorithm was giving accuracy of only 0.7 for the same forest fire dataset
