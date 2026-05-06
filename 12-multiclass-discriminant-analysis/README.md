Discriminant analysis for multiclass classification
================
Georgios Papadopoulos \|
2026-01-14

*Applying LDA, QDA and PCA assisted discriminant analysis to letter
recognition and high dimensional cancer classification*

## 1. Letter recognition with discriminant analysis

``` r
library(mlbench)
library(MASS)
library(ggplot2)

data(LetterRecognition)
#?LetterRecognition

train <- LetterRecognition[1:16000, ]
test  <- LetterRecognition[16001:20000, ]
```

### 1.1 Linear discriminant analysis on the test set

I set `newdata = test`

``` r
lda_fit <- lda(lettr ~ ., data = train)

lda_predict <- predict(lda_fit, newdata = test)$class
```

On the first look the confusion matrix shows correct input but it is not
helping because of its size with the whole alphabet. Therefore I did a
vizualization down below. Altogether the goal is to the misclassified
ones which are outside of the diagonal.

``` r
confusion_matrix <- table(True = test$lettr, Predicted = lda_predict)
confusion_matrix
```

    ##     Predicted
    ## True   A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R
    ##    A 126   1   0   0   0   0   0   1   0   6   1   0   2   0   3   0   0   1
    ##    B   0 100   0   1   0   2   1   4   1   0   1   0   0   0   2   0   0  13
    ##    C   0   0 102   0   6   1  12   2   0   0  11   0   1   1   1   0   0   0
    ##    D   0   4   0 139   0   0   2   1   2   1   0   0   1   1   3   0   0   3
    ##    E   0  10  19   1  70   1  20   0   0   0   0   0   0   0   0   0   2   3
    ##    F   0   5   0   5   0 104   4   1   0   0   0   0   0   1   0  19   4   1
    ##    G   1  15  31   1   3   0  73   4   0   0   8   0   2   0   4   0   6   7
    ##    H   1   0   0  10   0   1   2  67   0   0  13   0   2  13  16   2   3   7
    ##    I   0   1   1   2   2   2   1   0 136   2   0   0   0   0   0   3   3   0
    ##    J   0   1   0   3   0   5   0   1   9 110   0   0   0   0   2   2   3   0
    ##    K   0   5   1   3   5   0   5   0   0   0  90   0   1   1   3   0   1  22
    ##    L   3   7   3   0   3   0   8   0   2   5   3 112   0   0   0   0   2   1
    ##    M   1   0   0   0   0   0   0   3   0   0   1   0 125   3   0   0   0   3
    ##    N   1   0   0   4   0   0   0   8   0   0   2   0   5 133   1   0   0   0
    ##    O   4   2   1  14   0   0   2  22   0   1   1   0   0   2  87   0   1   0
    ##    P   0   3   0   6   0  12   1   0   1   0   1   0   0   0   4 120   5   2
    ##    Q   7   8   0   3   1   0  12   3   0   3   0   2   0   0  17   0  96   0
    ##    R   0  12   0  12   0   0   0   4   0   0  10   0   1   1   4   0   0 115
    ##    S   4  15   0   3   1   3   6   1   0   1   0   7   0   0   0   0   2   3
    ##    T   0   0   0   1   1  19   7   1   2   0   2   0   0   0   4   1   0   1
    ##    U   0   0   0   1   0   0   0  12   0   0   1   0   6   6   9   0   0   0
    ##    V   0   0   0   0   0   3   0   2   0   0   2   0   0   0   0   0   0   2
    ##    W   0   0   0   0   0   0   0   7   0   0   0   0   4   3   0   0   0   0
    ##    X   0   5   0   4   1   0   0   0   2   0   2   0   0   0   0   1   6   1
    ##    Y   0   0   0   4   0  16   0   0   0   0   0   0   0   0   0   0  13   0
    ##    Z   0   1   0   0   7   3   3   1   1   2   0   0   0   0   0   0   8   0
    ##     Predicted
    ## True   S   T   U   V   W   X   Y   Z
    ##    A   7   0   1   0   1   3   3   0
    ##    B   9   0   0   0   0   2   0   0
    ##    C   1   1   1   1   1   0   0   0
    ##    D   4   0   0   0   0   6   0   0
    ##    E   3   0   0   0   0  20   0   3
    ##    F   1   3   0   0   2   2   1   0
    ##    G   5   0   0   0   2   2   0   0
    ##    H   0   0   6   2   1   5   0   0
    ##    I   7   0   0   0   0   3   1   1
    ##    J   8   0   0   0   0   4   0   0
    ##    K   0   0   3   1   1   4   0   0
    ##    L   4   0   0   0   0   4   0   0
    ##    M   0   0   0   0   8   0   0   0
    ##    N   0   1   1   1   9   0   0   0
    ##    O   0   0   0   0   1   1   0   0
    ##    P   1   1   0   0   3   0   8   0
    ##    Q  10   0   0   0   3   2   1   0
    ##    R   0   0   0   0   0   2   0   0
    ##    S  76   0   0   2   0  15   1  21
    ##    T   3 104   0   1   0   2   0   2
    ##    U   0   0 131   0   0   2   0   0
    ##    V   0   0   1 119   6   1   0   0
    ##    W   0   0   0   0 125   0   0   0
    ##    X  11   1   5   0   0 118   0   2
    ##    Y   7   4   2  33   0   0  66   0
    ##    Z  16   0   0   0   0   7   0 109

Now it is very obvious to see the worst misclassified letters. I set the
frequency for the red tiles bigger than 15. I see in red that the
obvious ones are: Z with S Y with V K with R O with H G with C

It makes sense because of their similar shape.

``` r
ggplot(as.data.frame(confusion_matrix),
       aes(Predicted, True, fill = Freq)) +
  geom_tile(color = "white") +

  # highlight everything over 15
  geom_tile(
    data = subset(as.data.frame(confusion_matrix), Freq > 15 & True != Predicted),
    fill = "red",
    alpha = 0.6
  ) +

  scale_fill_gradient(low = "white", high = "black") +
  labs(
    title = "LDA confusion matrix",
    subtitle = "Cells with frequency > 15 highlighted in red",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text())
```

<img src="figures/LDA confusion matrix viz-1.png" style="display: block; margin: auto;" />

The misclassification rate is 31%, so I interpret it as moderate
classification performance. The correct classifications therefore are
69%.

``` r
misclass_rate <- mean(lda_predict != test$lettr)
misclass_rate
```

    ## [1] 0.31175

### 1.2 Fisher discriminant directions for letters

The first two Fisher discriminant directions visualize the projection
that maximizes the separation between each group relative to within
group variance.

Some letters form well separated clusters and are easy to distinguish
like: M W V N A L

The letters that looks similar have clusters that overlap like: Z with S
Y with V K with R O with H G with C

``` r
lda_fit <- lda(lettr ~ ., data = train)
lda_train <- predict(lda_fit, newdata = train)   

plot(lda_train$x[,1], lda_train$x[,2],
     type = "n",
     xlab = "LD1", ylab = "LD2",
     main = "Fisher LDA: first two discriminant directions")

text(lda_train$x[,1], lda_train$x[,2],
     labels = train$lettr,
     col = as.integer(train$lettr),
     cex = 0.6)
```

<img src="figures/Fishers discriminant direction-1.png" style="display: block; margin: auto;" />

### 1.3 Quadratic discriminant analysis on the test set

QDA shows a lower misclassification rate than LDA because it allows
class specific covariance matrices for each letter, whereas LDA assumes
equal covariance across all letters. This additional flexibility leads
to quadratic decision boundaries that better adapt to differences in
variance and correlation structures between letters.

The risk however is overfitting in small samples.

``` r
qda_fit <- qda(lettr ~ ., data = train)

qda_pred <- predict(qda_fit, newdata = test)$class

confusion_matrix_qda <- table(True = test$lettr, Predicted = qda_pred)
confusion_matrix_qda
```

    ##     Predicted
    ## True   A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R
    ##    A 147   0   0   2   0   0   0   0   0   1   0   0   0   0   0   0   0   0
    ##    B   0 122   0   1   0   0   0   3   1   0   1   0   0   0   0   0   0   5
    ##    C   0   0 119   0   4   1   4   0   0   0   6   1   2   0   4   0   0   0
    ##    D   0   3   0 155   0   0   0   0   0   0   0   0   1   0   2   1   0   1
    ##    E   0   1   1   0 123   2  10   1   0   0   1   3   0   0   0   0   0   0
    ##    F   0   2   0   0   1 134   2   0   1   0   0   0   1   4   0   5   0   0
    ##    G   0   0   9   2   0   1 141   0   0   0   1   0   1   0   2   0   2   1
    ##    H   0   1   1   7   0   3   3 104   0   2  11   0   4   0   5   0   0   4
    ##    I   0   0   0   4   1   5   0   0 136   4   3   0   0   0   0   0   0   1
    ##    J   0   0   0   1   0   1   0   2   2 135   0   0   1   0   1   0   2   1
    ##    K   0   0   4   1   0   0   3   2   0   0 125   0   0   0   0   0   1   7
    ##    L   0   1   0   1   1   0   2   2   0   0   1 135   0   0   0   0   6   1
    ##    M   0   1   0   0   0   0   2   0   0   0   1   0 137   0   0   0   0   0
    ##    N   0   0   0   5   0   1   0   3   0   0   4   0   2 140   2   0   0   3
    ##    O   2   0   0   3   0   0   2   3   0   0   0   0   0   0 122   0   4   0
    ##    P   0   1   0   3   0   6   0   0   0   0   1   0   0   0   0 148   2   2
    ##    Q   4   0   0   1   0   0   1   2   0   0   0   1   0   0   9   1 147   0
    ##    R   0   3   0   4   0   0   0   1   0   0   5   0   0   1   1   0   0 146
    ##    S   0   0   0   0   2   4   1   1   0   1   0   0   0   0   0   0   0   0
    ##    T   1   0   0   0   1   2   2   2   0   0   2   0   0   0   0   1   0   2
    ##    U   0   0   0   0   0   0   1   1   0   0   1   0   3   0   4   0   0   0
    ##    V   0   2   0   1   0   0   3   0   0   0   0   0   0   0   0   0   0   0
    ##    W   0   0   0   0   0   0   0   1   0   0   0   0   3   1   1   0   0   0
    ##    X   0   1   0   0   0   0   0   2   3   1   7   0   0   0   0   0   1   2
    ##    Y   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   6   0   0
    ##    Z   0   0   0   1   0   0   0   0   0   1   0   0   0   0   0   0   1   0
    ##     Predicted
    ## True   S   T   U   V   W   X   Y   Z
    ##    A   0   0   2   0   0   0   4   0
    ##    B   3   0   0   0   0   0   0   0
    ##    C   0   0   0   1   0   0   0   0
    ##    D   1   1   2   0   0   0   0   0
    ##    E   2   0   0   0   0   4   0   4
    ##    F   0   1   0   0   0   0   1   1
    ##    G   2   0   0   1   1   0   0   0
    ##    H   0   0   2   1   1   1   1   0
    ##    I   8   1   0   0   0   0   1   1
    ##    J   1   0   0   0   0   0   0   1
    ##    K   1   0   1   0   0   1   0   0
    ##    L   3   0   0   0   0   4   0   0
    ##    M   0   0   0   0   3   0   0   0
    ##    N   0   0   1   1   0   0   4   0
    ##    O   0   0   0   0   2   1   0   0
    ##    P   0   0   0   0   0   0   5   0
    ##    Q   2   0   0   0   0   0   0   0
    ##    R   0   0   0   0   0   0   0   0
    ##    S 145   2   0   0   0   0   0   5
    ##    T   1 132   1   0   0   1   3   0
    ##    U   0   0 152   0   6   0   0   0
    ##    V   0   0   0 124   6   0   0   0
    ##    W   0   0   0   0 133   0   0   0
    ##    X   4   1   1   0   0 135   1   0
    ##    Y   1   1   1  17   0   1 117   0
    ##    Z   4   2   1   0   0   1   1 146

Compared to 1a we can clearly see that for frequency bigger than 15
there is only one clear misclassification worst offender for letter Y
with V.

``` r
ggplot(as.data.frame(confusion_matrix_qda),
       aes(Predicted, True, fill = Freq)) +
  geom_tile(color = "white") +

  geom_tile(
    data = subset(as.data.frame(confusion_matrix_qda),
                  Freq > 15 & True != Predicted),
    fill = "red",
    alpha = 0.6
  ) +

  scale_fill_gradient(low = "white", high = "black") +
  labs(
    title = "QDA confusion matrix",
    subtitle = "Off-diagonal cells with frequency > 15 highlighted in red",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text())
```

<img src="figures/QDA confusion matrix viz-1.png" style="display: block; margin: auto;" />

This good classification above is further confirmed with the lower
misclassification rate of 12.5%.

``` r
misclass_rate_qda <- mean(qda_pred != test$lettr)
misclass_rate_qda
```

    ## [1] 0.125

### 1.4 Limitations of QDA for Fisher direction visualization

QDA prediction does not contain \$x Fisher’s linear discriminant
function scores.

Because QDA allows class-specific covariance matrices, there is no
common linear projection that maximizes the separation between letters.
Therefore a plot of the first two discriminant directions cannot be
produced in the same way as for LDA.

``` r
qda_pred <- predict(qda_fit, train)
summary(qda_pred)
```

    ##           Length Class  Mode   
    ## class      16000 factor numeric
    ## posterior 416000 -none- numeric

## 2. PCA-assisted LDA for high-dimensional cancer classification

The Khan dataset consists of training and test sets. Gene expression
levels for 2308 genes are available for 63 training subjects (xtrain),
with corresponding tumor labels (ytrain), and for 20 test subjects
(xtest, ytest).

``` r
library(ISLR)
data(Khan)
#?Khan
```

We varied the number of retained principal components from 1 to 62. For
each value of k, the training data were projected onto the first k
principal components and classified using linear discriminant analysis
with leave one out cross validation.

### 2.1 Selecting the number of principal components

Scaling is not needed because the data provided are genes that are
compared to each other.

``` r
pca_train <- prcomp(Khan$xtrain, center = TRUE, scale. = FALSE)

max_k <- 62
error_rate <- numeric(max_k)

for (k in 1:max_k) {
  
  Xk <- pca_train$x[, 1:k, drop = FALSE]
  
  lda_cv <- lda(Xk, grouping = Khan$ytrain, CV = TRUE)
  
  error_rate[k] <- mean(lda_cv$class != Khan$ytrain)
}
```

The misclassification rate is high for very small values of k, decreases
for intermediate values, and increases again as k approaches the
maximum. Too few components lead to underfitting, while too many
components reintroduce noise and unstable covariance estimation. The
lowest error is achieved for k=10 components.

``` r
plot(1:max_k, error_rate, pch=20,
     xlab = "Number of principal components k",
     ylab = "Leave one out error rate",
     main = "Number of PCs")
```

<img src="figures/plot for PCS-1.png" style="display: block; margin: auto;" />

### 2.2 LDA in the selected PCA space

With Optimal k = 10, I use Xtrain_k that contains the scores of the
training obs for the first 10 PCs. Then I project the test data into the
same reduced PCA space learned from the training data for the same first
10 PCs. So now they are on the same PCA space.

Since ytrain are the the four tumor types, by fitting LDA on the same
PCA space each patient is described by k numbers instead of 2308 genes
which helps to separate the tumor classes.

``` r
k_optimal <- which.min(error_rate)

Xtrain_k <- pca_train$x[, 1:k_optimal, drop = FALSE]
#dim(Xtrain_k) #63 patients with 10 PCs

Xtest_k <- predict(pca_train, newdata = Khan$xtest)[, 1:k_optimal, drop = FALSE]
#dim(Xtest_k) #20 patients with 10 PCs

lda_k <- lda(x = Xtrain_k, grouping = Khan$ytrain)

test_pred <- predict(lda_k, newdata = Xtest_k)$class
```

### 2.3 Test set confusion table and misclassification error

The confusion table reveals only little misclassified test observation,
resulting in a misclassification error of 20%.

``` r
tab_test <- table(True = Khan$ytest, Predicted = test_pred)
tab_test
```

    ##     Predicted
    ## True 1 2 3 4
    ##    1 3 0 0 0
    ##    2 0 5 0 1
    ##    3 0 0 3 3
    ##    4 0 0 0 5

``` r
test_error <- mean(test_pred != Khan$ytest)
test_error
```

    ## [1] 0.2

### 2.4 Fisher discriminant visualization for training and test data

The Fisher discriminant plots show that the tumor groups are well
separated in the discriminant space for the training data.

``` r
lda_train_scores <- predict(lda_k)$x

plot(lda_train_scores[,1], lda_train_scores[,2],
     col = Khan$ytrain,
     pch = 19,
     xlab = "LD1",
     ylab = "LD2",
     main = "Fisher discriminant functions with training data")
legend("topright", legend = levels(as.factor(Khan$ytrain)),
       col = 1:4, pch = 19)
```

<img src="figures/train - fisher discriminant plot-1.png" style="display: block; margin: auto;" />
We see a similar pattern for the test data that shows that the LDA
classifier generalizes well to unseen observations.

``` r
lda_test_scores <- predict(lda_k, newdata = Xtest_k)$x

plot(lda_test_scores[,1], lda_test_scores[,2],
     col = Khan$ytest,
     pch = 19,
     xlab = "LD1",
     ylab = "LD2",
     main = "Fisher discriminant functions with test data")
legend("topright", legend = levels(as.factor(Khan$ytest)),
       col = 1:4, pch = 19)
```

<img src="figures/test - fisher discriminant plot-1.png" style="display: block; margin: auto;" />
