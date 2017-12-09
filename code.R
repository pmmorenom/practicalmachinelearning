# PRACTICAL MACHINE LEARNING
# PEDRO MANUEL MORENO MARCOS
library(caret)
library(randomForest)

training <- read.csv('./pml-training.csv', header=TRUE, na.strings=c("NA","#DIV/0",""))
testing <- read.csv('./pml-testing.csv', header=TRUE, na.strings=c("NA","#DIV/0",""))

na_count <-sapply(training, function(y) sum(length(which(is.na(y)))))
training2 = training[na_count < dim(training)[1]*0.5]

inTrain = createDataPartition(training2$classe, p = 0.6)[[1]]
tr = training2[ inTrain,]
val = training2[-inTrain,]

preObj <- preProcess(tr[,-dim(tr)[2]],method="knnImpute")
tr2 <- predict(preObj, tr[,-dim(tr)[2]])
tr2 = data.frame(tr2[,8:59], classe=tr$classe)

train_control <- trainControl(method="cv", number=10)
modFit <- randomForest(classe ~ ., data=tr2, trainControl=train_control)

val2 <- predict(preObj, val[,-dim(val)[2]])
val2 = data.frame(val2[,8:59], classe=val$classe)
predVal = predict(modFit, val2)
confusionMatrix(val$classe, predVal)

testing2 = testing[na_count < dim(training)[1]*0.5]
testing2 = predict(preObj, testing2[,-dim(tr)[2]])
output = predict(modFit, testing2[,8:59])