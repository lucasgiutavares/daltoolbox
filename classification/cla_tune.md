## Tune Regression 


```r
# DAL ToolBox
# version 1.1.727



#loading DAL
library(daltoolbox) 
```

### Dataset for classification


```r
iris <- datasets::iris
head(iris)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
## 1          5.1         3.5          1.4         0.2  setosa
## 2          4.9         3.0          1.4         0.2  setosa
## 3          4.7         3.2          1.3         0.2  setosa
## 4          4.6         3.1          1.5         0.2  setosa
## 5          5.0         3.6          1.4         0.2  setosa
## 6          5.4         3.9          1.7         0.4  setosa
```

```r
#extracting the levels for the dataset
slevels <- levels(iris$Species)
slevels
```

```
## [1] "setosa"     "versicolor" "virginica"
```

## Building samples (training and testing)


```r
# preparing dataset for random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

tbl <- rbind(table(iris[,"Species"]),
             table(iris_train[,"Species"]),
             table(iris_test[,"Species"]))
rownames(tbl) <- c("dataset", "training", "test")
head(tbl)
```

```
##          setosa versicolor virginica
## dataset      50         50        50
## training     39         38        43
## test         11         12         7
```

### Training


```r
tune <- cla_tune(cla_svm("Species", slevels))
ranges <- list(epsilon=seq(0,1,0.2), cost=seq(20,100,20), kernel = c("linear", "radial", "polynomial", "sigmoid"))

model <- fit(tune, iris_train, ranges)
```

### Model adjustment


```r
train_prediction <- predict(model, iris_train)

iris_train_predictand <- adjust_class_label(iris_train[,"Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9833333 39 81  0  0         1      1           1           1  1
```

### Test


```r
# Test
test_prediction <- predict(model, iris_test)

iris_test_predictand <- adjust_class_label(iris_test[,"Species"])

#Avaliação #setosa
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9333333 11 19  0  0         1      1           1           1  1
```

```r
#Avaliação #versicolor
test_eval <- evaluate(model, iris_test_predictand, test_prediction, ref=2)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity        f1
## 1 0.9333333 12 16  2  0 0.8571429      1           1   0.8888889 0.9230769
```

```r
#Avaliação #virginica
test_eval <- evaluate(model, iris_test_predictand, test_prediction, ref=3)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision    recall sensitivity specificity        f1
## 1 0.9333333  5 23  0  2         1 0.7142857   0.7142857           1 0.8333333
```

### Options for other models


```r
#knn
ranges <- list(k=1:20)

#mlp
ranges <- list(size=1:10, decay=seq(0, 1, 0.1))

#rf
ranges <- list(mtry=1:3, ntree=1:10)
```
