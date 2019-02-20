#Comparing Gaussian Naive Bayes and Logistic Regression algorithms on ecoli data set

## Read the data
xTrain=as.matrix(read.csv("ecoli_xTrain.csv", header=FALSE))
yTrain=as.matrix(read.csv("ecoli_yTrain.csv", header=FALSE))
xTest=as.matrix(read.csv("ecoli_xTest.csv", header=FALSE))
yTest=as.matrix(read.csv("ecoli_yTest.csv", header=FALSE))

prior <- function(yTrain){
  classes = sort(unlist(unique(yTrain)))
  p = c()
  
  for (ind in 1:length(classes)){
    p[ind] = (length(which(yTrain==classes[ind]))/ nrow(yTrain))
    
  }
  
  return(t(t(p))) 
  
}

likelihood <- function(xTrain, yTrain){
  label = sort(unlist(unique(yTrain))) #classes
  nlabel = length(label) 
  nfeature = ncol(xTrain)
  nTrain_points = nrow(xTrain)
  mu = matrix(0, nrow = nfeature, ncol = nlabel) #creating empty matrix for conditional mean
  sigma = matrix(0, nrow = nfeature, ncol = nlabel) #creating empty matrix for conditional variance
  for (each_label in 1:nlabel){
    for (each_feat in 1:nfeature){
      x = xTrain[,each_feat][yTrain == label[each_label]] #returns data points of feature i given class j
      mu[each_feat,each_label] = mean(x)
      sigma[each_feat, each_label] = sd(x)
    }  
  }
  return(list(mu,sigma)) 
}


naiveBayesClassify <- function(xTest, M, V, p){
  label = sort(unlist(unique(yTrain))) #classes
  nlabel = length(label) 
  nfeature = ncol(xTrain)
  nTrain = nrow(xTrain)
  nTest = nrow(xTest)
  y_pred = matrix(0, nrow=nTest, ncol = 1)
  
#  p = prior(yTrain)
#  M = likelihood(xTrain, yTrain)[[1]]
#  V = likelihood(xTrain, yTrain)[[2]]
  
  for (m in 1:nTest){
    prob = p #make another copy of the prior so could update it later
    for (each_label in 1:nlabel){
      for (each_feat in 1:nfeature){
        mu = M[each_feat,each_label]
        sigma = V[each_feat,each_label]
        prob[each_label] = prob[each_label] * (dnorm(xTest[m,each_feat],  mean=mu, sd=sigma, log=FALSE)) #classifying Xnew
      }
    }
    y_pred[m] = label[which.max(prob)]
    
  }
  
  return(y_pred)
}

p = prior(yTrain)
M = likelihood(xTrain,yTrain)[[1]]
V = likelihood(xTrain, yTrain)[[2]]
ypred = naiveBayesClassify(xTest,M,V,p)


##Testing results ####
#Fraction of test samples classified correctly
accuracy_fraction <- function(yTest , ypred){
  correct = length(yTest[yTest == ypred])
  correct_fraction = correct / dim(yTest)[1]
  return(correct_fraction)
}
correct_fraction_p2 = accuracy_fraction(yTest,ypred) #0.8345

#Precision for class 1; TP/TP+FP
precision <- function(class,yTest,ypred){
  same = yTest[yTest==ypred]
  TP = length(same[same==class]) #predicted 1, actual 1
  pred = which(ypred==class)
  compare = yTest[ypred==class] #shows yTest values of when y_pred is 1, so can see any different values
  FP = length(compare[compare != class]) #actually not 1
  precision_out = TP / (TP + FP)
  return(precision_out)
}
precision1_p2 = precision(1,yTest,ypred) #0.9512

#Recall for class 1; TP / TP+FN
recall <- function(class, yTest, ypred){
  same = yTest[yTest==ypred]
  TP = length(same[same==class])
  diff = yTest[yTest != ypred]
  FN = length(diff[diff==class])
  recall_out = TP/ (TP+FN)
  return(recall_out)
}
recall1_p2 = recall(1,yTest,ypred) #Recall; 0.975

#Precision for class 5
precision5_p2 = precision(5,yTest,ypred) #0.875

#Recall for class 5
recall5_p2 = recall(5,yTest,ypred) #0.778


#########################

sigmoidProb <- function(y, x, w){
  if (y==0){
    p = 1 / (1 + exp(x %*% w))
    
  }
  else {
    p = exp(x %*% w) / (1 + exp(x %*% w))
    
  }
  return(p)
}

logisticRegressionWeights <- function(xTrain, yTrain, w0, nIter){
  for (i in 1:nIter){
    #p = exp(xTrain %*% w0) / (1+ exp(xTrain %*% w0))
    p = sigmoidProb(1,xTrain,w0)
    gradient = t((yTrain - p)) %*% xTrain
    #gradient = (t(yTrain) %*% xTrain) - (t(p) %*% xTrain)
    alpha = 0.1
    w0 = w0 + (alpha * t(gradient))
  }
  return(w0)
}

logisticRegressionClassify <- function(xTest, w){
  #p = 1 / (1 + exp(xTest %*% w))
  pred = matrix(0, nrow=nrow(xTest), ncol=1)
  pred0 = sigmoidProb(0, xTest, w)
  pred1 = sigmoidProb(1, xTest, w)
  
  pred[pred0>=pred1]=0
  pred[pred0<=pred1]=1
  return(pred)
  
}

w0 = matrix(0.1, nrow=6, ncol=1) #initial weights
xTrain_new=as.matrix(read.csv("ecoli_new.xTrain.csv", header=FALSE))
yTrain_new= as.matrix(read.csv("ecoli_new.yTrain.csv", header=FALSE))
xTest_new=as.matrix(read.csv("ecoli_new.xTest.csv", header=FALSE))
yTest_new=as.matrix(read.csv("ecoli_new.yTest.csv", header=FALSE))

w_best = logisticRegressionWeights(xTrain_new, yTrain_new,w0, 20000)
ypred_new = logisticRegressionClassify(xTest_new,w_best)

# Evaluation
## Testing results for Logistic Regression ####
#Fraction of test samples classified correctly
correct_fraction_p3_LR = accuracy_fraction(yTest_new, ypred_new) #0.954

#Precision for class 1; TP/TP+FP
precision_p3_LR = precision(1,yTest_new,ypred_new) #0.973

#Recall for class 1; TP / TP+FN
recall_p3_LR = recall(1,yTest_new,ypred_new) #0.9


###Testing for GNB on new ecoli data set

naiveBayesClassifynew <- function(xTest, M, V, p){
  label = sort(unlist(unique(yTrain_new))) #classes
  nlabel = length(label) 
  nfeature = ncol(xTrain_new)
  nTrain = nrow(xTrain_new)
  nTest = nrow(xTest)
  y_pred = matrix(0, nrow=nTest, ncol = 1)

  for (m in 1:nTest){
    prob = p #make another copy of the prior so could update it later
    for (each_label in 1:nlabel){
      for (each_feat in 1:nfeature){
        mu = M[each_feat,each_label]
        sigma = V[each_feat,each_label]
        prob[each_label] = prob[each_label] * (dnorm(xTest[m,each_feat],  mean=mu, sd=sigma, log=FALSE)) #classifying Xnew
      }
    }
    y_pred[m] = label[which.max(prob)]
  }
  return(y_pred)
}

p_newGNB = prior(yTrain_new)
M_newGNB = likelihood(xTrain_new,yTrain_new)[[1]]
V_newGNB = likelihood(xTrain_new, yTrain_new)[[2]]
ypred_newGNB = naiveBayesClassifynew(xTest_new,M_newGNB,V_newGNB,p_newGNB)

#Evaluation of GNB on new Ecoli Data set 
correct_fraction_newGNB = accuracy_fraction(yTest_new, ypred_newGNB) #0.633
precision_p3_newGNB = precision(1,yTest_new,ypred_newGNB) #NAN
recall_p3_newGNB = recall(1,yTest_new,ypred_newGNB) #0
