# gnblr
Simple Gaussian Naive Bayes &amp; Logistic Regression implementation

For GNB, use ecoli_xTest.csv, ecoli_xTrain.csv, ecoli_yTest.csv, ecoli_yTrain.csv.

For LR, use ecoli_new.xTest.csv, ecoli_new.xTrain.csv, ecoli_new.yTest.csv, ecoli_new.yTrain.csv.

In the GNB training-testing dataset, ecoli has 5 classes; while in the LR training-testing dataset, ecoli_new has only 2 classes.

GNB evaluation
--------------
0.835 #Fraction of test samples classified correctly<br />
0.951 #Precision for class 1<br />
0.975 #Recall for class 1<br />
0.875 #Precision for class 5<br />
0.778 # Recall for class 5

#Evaluation of ecoli_new dataset<br />
--------------
Using LR

0.954 #Fraction of test samples classified correctly<br />
0.973 #Precision for class 1<br />
0.9 #Recall for class 1<br />

Using GNB

0.633 #Fraction of Test samples classified correctly<br />
Nan #Precision for class 1<br />
0 #Recall for class 1


Here, we can see that Logistic Regression (LR) performs significantly better than Gaussian Naive Bayes (GNB). This could be due X1,...,Xn are not following Gaussian distribution and are conditionally dependent. In the ecoli_new.xTrain.csv dataset, the first column of values are all ones, which could make the GNB fail to learn the parameters properly.

