# Clustering - pam
## Libraries and Datasets


```r
# DAL ToolBox
# version 1.1.727



#loading DAL
library(daltoolbox) 
```


```r
#load dataset
data(iris)
```

General entropy of dataset

## General function to test clustering methods


```r
# setup clustering
model <- cluster_pam(k=3)
```


```r
# build model
model <- fit(model, iris[,1:4])
clu <- cluster(model, iris[,1:4])
table(clu)
```

```
## clu
##  1  2  3 
## 50 62 38
```


```r
# evaluate model using external metric
eval <- evaluate(model, clu, iris$Species)
eval
```

```
## $clusters_entropy
## # A tibble: 3 × 4
##   x        ce   qtd    ceg
##   <fct> <dbl> <int>  <dbl>
## 1 1     0        50 0     
## 2 2     0.771    62 0.319 
## 3 3     0.297    38 0.0754
## 
## $clustering_entropy
## [1] 0.3938863
## 
## $data_entropy
## [1] 1.584963
```

