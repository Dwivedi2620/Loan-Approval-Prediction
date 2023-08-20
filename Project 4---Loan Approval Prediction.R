
#-----------------------Project 4 – Loan Approval Prediction-----------------------#

library(dplyr)
library(Matrix)
library(ggplot2)
library(caTools)
library(e1071)
library(caret)


# A) Data Preprocessing:
 
# a. Have a glance at the structure of the dataset and find if there are any missing values present


Customer_loan<-read.csv("G:/My Drive/Data Science with R/Data Sets/customer_loan.csv",stringsAsFactors = T)
colSums(is.na(Customer_loan))
View(Customer_loan)



# b. Calculate the debt-to-income ratio and add it as a new column named ‘dti’


Customer_loan %>% mutate(dti= (debts/income)) -> Customer_loan



# c. Create a new variable named ‘loan_decision_status’, where the value would b ‘0’ if ‘loan_decision_type’ is equal to ‘denied’, else it would be ‘1’


Customer_loan %>% mutate(loan_decision_status=
                         ifelse( tolower(loan_decision_type)=="denied",0,1))  -> Customer_loan



# i. Convert this variable into a factor


Customer_loan$loan_decision_status <- as.factor(Customer_loan$loan_decision_status)



# d. Create a new data-set named ‘customer_loan_refined’, which would have these column numbers from the original dataframe - (3,4,6,7,8,11,13,14)


customer_loan_refined <- Customer_loan[,c(3,4,6,7,8,11,13,14)]



# e. Encode ‘gender’, ‘marital_status’, ‘occupation’, and ‘loan_type’ as factors and then convert them into numeric 


customer_loan_refined$gender   <- as.numeric(as.factor(customer_loan_refined$gender))-1
customer_loan_refined$marital_status <- as.numeric(as.factor(customer_loan_refined$marital_status))-1
customer_loan_refined$occupation<- as.numeric(as.factor(customer_loan_refined$occupation))-1
customer_loan_refined$loan_type<- as.numeric(as.factor(customer_loan_refined$loan_type))-1



#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------




# B) Model Building:


# a. Divide the data into ‘train’ & ‘test’ sets and set the split-ratio to be 70%


sample_tag <- sample.split(customer_loan_refined$loan_decision_status,SplitRatio = 0.7)
train <- subset(customer_loan_refined,sample_tag)
test <- subset(customer_loan_refined,sample_tag==F)



# b. Apply feature scaling on all the columns of ‘train’ & ‘test’ set, except the ‘loan_decision_status’ column


traintest_combined <- rbind(train,test)

principal_component <- prcomp(traintest_combined[,-8], scale. = T)

summary(principal_component)

plot(principal_component)
names(principal_component)
principal_component$center
principal_component$rotation[1:5,]
biplot(principal_component, scale = 0)

train.data <- data.frame(loan_decision_status=traintest_combined$loan_decision_status,
                         principal_component$x)



# c. Apply principal component analysis on the first 7 columns of ‘train’ & ‘test’ set. The number of principal components obtained should be 2


train.data <- train.data[,1:3]

train.data$loan_decision_status <- as.factor(train.data$loan_decision_status)



# d. Build the naïve bayes model on the train set


sample_tag<- sample.split(train.data$loan_decision_status,SplitRatio = 0.7)
trainm <- subset(train.data,sample_tag)
testm <- subset(train.data,sample_tag==F)

Naive_Bayes_Model <- naiveBayes(trainm[,-1],trainm$loan_decision_status)
summary(Naive_Bayes_Model)


# e. Predict the values on the test set


prediction <-predict(Naive_Bayes_Model,newdata = testm)


# f. Build a confusion matrix for actual values and predicted values


confusionMatrix(prediction,testm$loan_decision_status)




########################################################################################################################################################