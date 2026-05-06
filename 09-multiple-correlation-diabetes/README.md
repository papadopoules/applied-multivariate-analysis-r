Multiple correlation analysis of diabetes biomarkers
================
Georgios Papadopoulos \|
2025-12-01

*Assessing relationships between relative weight and glucose-insulin
features across diagnostic groups*

# 1. Multiple correlation by diagnostic group

The Diabetes dataset consists of p=6 variables and n=145 obs, out of
which the ones related with glucose plasma are assigned onto
`gluc.vars`. There are three levels of groups, namely Normal,
Chemical_Diabetic and Overt_Diabetic.

``` r
library(heplots)

data("Diabetes")
#?Diabetes

str(Diabetes)
```

    ## 'data.frame':    145 obs. of  6 variables:
    ##  $ relwt  : num  0.81 0.95 0.94 1.04 1 0.76 0.91 1.1 0.99 0.78 ...
    ##  $ glufast: int  80 97 105 90 90 86 100 85 97 97 ...
    ##  $ glutest: int  356 289 319 356 323 381 350 301 379 296 ...
    ##  $ instest: int  124 117 143 199 240 157 221 186 142 131 ...
    ##  $ sspg   : int  55 76 105 108 143 165 119 105 98 94 ...
    ##  $ group  : Factor w/ 3 levels "Normal","Chemical_Diabetic",..: 1 1 1 1 1 1 1 1 1 1 ...

``` r
summary(Diabetes)
```

    ##      relwt           glufast       glutest          instest     
    ##  Min.   :0.7100   Min.   : 70   Min.   : 269.0   Min.   : 10.0  
    ##  1st Qu.:0.8800   1st Qu.: 90   1st Qu.: 352.0   1st Qu.:118.0  
    ##  Median :0.9800   Median : 97   Median : 413.0   Median :156.0  
    ##  Mean   :0.9773   Mean   :122   Mean   : 543.6   Mean   :186.1  
    ##  3rd Qu.:1.0800   3rd Qu.:112   3rd Qu.: 558.0   3rd Qu.:221.0  
    ##  Max.   :1.2000   Max.   :353   Max.   :1568.0   Max.   :748.0  
    ##       sspg                     group   
    ##  Min.   : 29.0   Normal           :76  
    ##  1st Qu.:100.0   Chemical_Diabetic:36  
    ##  Median :159.0   Overt_Diabetic   :33  
    ##  Mean   :184.2                         
    ##  3rd Qu.:257.0                         
    ##  Max.   :480.0

``` r
gluc.vars <- c("glufast", "glutest", "instest", "sspg")
groups <- levels(Diabetes$group)
```

To calculate r for each group I wrote a function that calculates the
correlation matrix for each group from page 100/126.

$$
r^{2}_{x,y} = R^{\top}_{yx} \, R^{-1}_{yy} \, R_{yx}
$$

Where x = relwt and y = (glufast, glutest, instest, sspg)

``` r
compute_multcorr <- function(g) {
  group_data <- subset(Diabetes, group == g)
  
  x <- group_data$relwt
  y <- group_data[, gluc.vars]
  
  Ryy <- cor(y)
  Ryx <- cor(y, x)
  
  r2_value <- as.numeric(t(Ryx) %*% solve(Ryy) %*% Ryx)
  r_value  <- sqrt(r2_value)
  
  data.frame(group = g, r2 = r2_value, r = r_value)
}

ex1_results <- do.call(rbind, lapply(groups, compute_multcorr))
```

Across the diagnostic groups, the multiple correlation increases from
0.55 in the normal group to 0.56 in the chemical diabetic group and
reaches 0.72 in the overt diabetic group. This means that body weight is
much more closely related to glucose levels in people with more advanced
diabetes.

``` r
ex1_results
```

    ##               group        r2         r
    ## 1            Normal 0.3024183 0.5499258
    ## 2 Chemical_Diabetic 0.3132089 0.5596507
    ## 3    Overt_Diabetic 0.5150899 0.7176976

# 2. Interpretation of linear predictor coefficients

The coefficients of the linear predictor $a_{0}+a^{\top}y$ can be
interpreted in the same way as regression coefficients. Each coefficient
describes how the predicted relative weight `relwt` changes when one
glucose plasma variable increases, while the other variables are kept
fixed. A positive coefficient means that higher values of this variable
are associated with higher predicted relative weight `relwt` and a
negative coefficient means the opposite. The larger the absolute value
of a coefficient, the more important that variable is in the linear
prediction of `relwt`. we can see which glucose variables are most
influential in each group and how their influence changes with group
severity by comparing these coefficients between groups.

Because I was not sure from the exercise instruction whether I needed to
compute the coefficients, I did it depending on each of the three groups
as we saw for sample data in the lecture with $a=R^{-1}_{yy}\,R_{yx}$
where we see for example that for every group the `sspg` variable has
the most influence on each group with a positive coeff.

``` r
compute_coeffs <- function(g) {
  group_data <- subset(Diabetes, group == g)

  x <- group_data$relwt
  y <- group_data[, gluc.vars]

  Ryy <- cor(y)
  Ryx <- cor(y, x)

  a  <- solve(Ryy) %*% Ryx
  rownames(a) <- gluc.vars

  data.frame(group = g, variable = gluc.vars, coefficient = as.numeric(a))
}

coeffs_ex2 <- do.call(rbind, lapply(groups, compute_coeffs))

coeffs_ex2
```

    ##                group variable coefficient
    ## 1             Normal  glufast  0.23086566
    ## 2             Normal  glutest  0.08279681
    ## 3             Normal  instest -0.22233343
    ## 4             Normal     sspg  0.51516803
    ## 5  Chemical_Diabetic  glufast  0.16637975
    ## 6  Chemical_Diabetic  glutest -0.34211534
    ## 7  Chemical_Diabetic  instest -0.07915052
    ## 8  Chemical_Diabetic     sspg  0.49681457
    ## 9     Overt_Diabetic  glufast  0.46790255
    ## 10    Overt_Diabetic  glutest -1.33625129
    ## 11    Overt_Diabetic  instest -0.16568910
    ## 12    Overt_Diabetic     sspg  0.72778339

# 3. Significance of multiple correlations

To test whether the multiple correlation is zero, we use the F-test

$$
F = \frac{(n - 1 - p)\, r^{2}_{x,y}}{p(1 - r^{2}_{x,y})}
$$

with p=4 predictors and n=145 obs.

Codewise, I do the same function as in task 1 but now I also calculate
Fstat and the right tailed probability.

``` r
compute_test <- function(g) {
  group_data <- subset(Diabetes, group == g)

  x <- group_data$relwt
  y <- group_data[, gluc.vars]

  Ryy <- cor(y)
  Ryx <- cor(y, x)

  r2 <- as.numeric(t(Ryx) %*% solve(Ryy) %*% Ryx)
  r  <- sqrt(r2)

  n  <- nrow(y)
  p  <- ncol(y)

  Fstat <- (n - 1 - p) * r2 / (p * (1 - r2))
  pval  <- 1 - pf(Fstat, p, n - 1 - p)

  data.frame(
    group = g,
    r = r,
    r2 = r2,
    F = Fstat,
    p.value = pval
  )
}

ex3_results <- do.call(rbind, lapply(groups, compute_test))
```

Although a relatively large $r^2$ already suggests that the multiple
correlation is likely to be significant, statistical significance cannot
be determined from $r^2$ alone, because it also depends on the sample
size and the number of predictors. The Ftest confirms that all three
$r^2$ values are significantly different from zero, as all p-values fall
below the 0.05 threshold. Therefore, we reject the null hypothesis that
the glucose plasma variables are jointly uncorrelated with weight in
every group.

``` r
ex3_results
```

    ##               group         r        r2        F      p.value
    ## 1            Normal 0.5499258 0.3024183 7.695050 3.289569e-05
    ## 2 Chemical_Diabetic 0.5596507 0.3132089 3.534364 1.730969e-02
    ## 3    Overt_Diabetic 0.7176976 0.5150899 7.435665 3.263375e-04

# 4. Pearson CCA grid comparison

Same results with CCAgrid. This happens because we only have one
variable in X = `relwt`. In that case, the canonical correlation is the
same thing as the multiple correlation. So CCAgrid gives the same
numbers as before.

``` r
library(ccaPP)
#?CCAgrid

# Subset group 1
g1 <- subset(Diabetes, group == "Normal")

X <- as.matrix(g1$relwt)
Y <- as.matrix(g1[, gluc.vars])

res1 <- CCAgrid(X, Y, k = 1, method = "pearson", standardize = TRUE)

# Subset group 2
g2 <- subset(Diabetes, group == "Chemical_Diabetic")

X <- as.matrix(g2$relwt)
Y <- as.matrix(g2[, gluc.vars])

res2 <- CCAgrid(X, Y, k = 1, method = "pearson", standardize = TRUE)

# Subset group 3
g3 <- subset(Diabetes, group == "Overt_Diabetic")

X <- as.matrix(g3$relwt)
Y <- as.matrix(g3[, gluc.vars])

res3 <- CCAgrid(X, Y, k = 1, method = "pearson", standardize = TRUE)

ex4_results <- c(normal = res1$cor, chemical = res2$cor, overt = res3$cor)
ex4_results
```

    ##    normal  chemical     overt 
    ## 0.5499257 0.5596506 0.7167311

# 5. Spearman CCA grid comparison

method = “spearman” produces slightly higher correlations than Pearson,
but the overall pattern remains the same. The methodological difference
is that Spearman correlation is rank based and captures also non linear
relationships, while Pearson correlation measures linear relationships
using the raw data. Spearman is therefore less sensitive to outliers,
which I believe explains the small increase in the correlation values.

``` r
# Subset group 1
g1 <- subset(Diabetes, group == "Normal")

X <- as.matrix(g1$relwt)
Y <- as.matrix(g1[, gluc.vars])

res1 <- CCAgrid(X, Y, k = 1, method = "spearman", standardize = TRUE)

# Subset group 2
g2 <- subset(Diabetes, group == "Chemical_Diabetic")

X <- as.matrix(g2$relwt)
Y <- as.matrix(g2[, gluc.vars])

res2 <- CCAgrid(X, Y, k = 1, method = "spearman", standardize = TRUE)

# Subset group 3
g3 <- subset(Diabetes, group == "Overt_Diabetic")

X <- as.matrix(g3$relwt)
Y <- as.matrix(g3[, gluc.vars])

res3 <- CCAgrid(X, Y, k = 1, method = "spearman", standardize = TRUE)

ex5_results <- c(normal = res1$cor, chemical = res2$cor, overt = res3$cor)
ex5_results
```

    ##    normal  chemical     overt 
    ## 0.5636177 0.5998729 0.7737556

# 6. Effect of transformations on rank-based results

Spearman correlation is based on ranks, not on the original numerical
values. Therefore, monotone transformations such as the logarithm should
not change the rank order of the observations and should not change the
Spearman correlations.

In this setting, `method = "spearman"` applies the CCA procedure to
rank-based information. Since the log transformation preserves the
ordering of the variables, the results should remain the same or nearly
identical, apart from possible numerical differences.

Therefore, transforming the variables before applying Spearman-based CCA
is generally unnecessary. Transformations are more relevant for
Pearson-based methods, where the actual values, scale, skewness, and
outliers directly affect the estimated correlations.

``` r
Diabetes_log <- Diabetes
Diabetes_log$relwt   <- log(Diabetes$relwt + 1)
Diabetes_log$glufast <- log(Diabetes$glufast + 1)
Diabetes_log$glutest <- log(Diabetes$glutest + 1)
Diabetes_log$instest <- log(Diabetes$instest + 1)
Diabetes_log$sspg    <- log(Diabetes$sspg + 1)

g1 <- subset(Diabetes_log, group == "Normal")
X1 <- as.matrix(g1$relwt)
Y1 <- as.matrix(g1[, c("glufast","glutest","instest","sspg")])

res1 <- CCAgrid(X1, Y1, k = 1, method = "spearman", standardize = TRUE)
res1$cor
```

    ## [1] 0.5652533

``` r
g2 <- subset(Diabetes_log, group == "Chemical_Diabetic")
X2 <- as.matrix(g2$relwt)
Y2 <- as.matrix(g2[, c("glufast","glutest","instest","sspg")])

res2 <- CCAgrid(X2, Y2, k = 1, method = "spearman", standardize = TRUE)
res2$cor
```

    ## [1] 0.6019322

``` r
g3 <- subset(Diabetes_log, group == "Overt_Diabetic")
X3 <- as.matrix(g3$relwt)
Y3 <- as.matrix(g3[, c("glufast","glutest","instest","sspg")])

res3 <- CCAgrid(X3, Y3, k = 1, method = "spearman", standardize = TRUE)
res3$cor
```

    ## [1] 0.6385748

# 7. Permutation test for uncorrelatedness

permTest does a permutation test for uncorrelatedness by repeatedly
shuffling the rows of the first input X. Each shuffle breaks any real
relationship between X and Y. After each permutation, the function
creates a new canonical correlation. These correlations represent what
the correlation would look like if X and Y were unrelated. Then we
compare the real correlation to these fake ones basically. If the real
correlation is bigger than all of the permuted ones, it means the
relationship is real and not just random. If X and Y were truly
unrelated, shuffling X would not change anything. A permutation test
does not rely on a distributional assumption.

# 8. Pearson permutation test results

The permutation test shows that all three canonical correlations are
statistically significant.

- For the Normal group, the p-value is essentially 0, indicating very
  strong evidence of association.

- For the Chemical Diabetic group, the p-value is 0.03, which is still
  below 0.05, so the association is also significant, although weaker.

- For the Overt Diabetic group, the p-value is again effectively 0,
  showing a very strong relationship.

These results match the earlier F tests from Exercise 3: in every group,
the glucose plasma variables are significantly related to `relwt`.

``` r
#?permTest
# subset group 1
g1 <- subset(Diabetes, group == "Normal")

X1 <- as.matrix(g1$relwt)             # first set
Y1 <- as.matrix(g1[, gluc.vars])      # second set

set.seed(123)
perm_normal <- permTest(X1, Y1, R = 1000, method="pearson")
perm_normal
```

    ## 
    ## Permutation test for no association
    ## 
    ## r = 0.549926, p-value = 0.000000
    ## R = 1000 random permuations
    ## Alternative hypothesis: true maximum correlation is not equal to 0

``` r
# subset group 2
g2 <- subset(Diabetes, group == "Chemical_Diabetic")
X2 <- as.matrix(g2$relwt)
Y2 <- as.matrix(g2[, gluc.vars])

set.seed(123)
perm_chemical <- permTest(X2, Y2, R = 1000, method="pearson")
perm_chemical
```

    ## 
    ## Permutation test for no association
    ## 
    ## r = 0.559651, p-value = 0.015000
    ## R = 1000 random permuations
    ## Alternative hypothesis: true maximum correlation is not equal to 0

``` r
# subset group 3
g3 <- subset(Diabetes, group == "Overt_Diabetic")
X3 <- as.matrix(g3$relwt)
Y3 <- as.matrix(g3[, gluc.vars])

set.seed(123)
perm_overt <- permTest(X3, Y3, R = 1000, method="pearson")
perm_overt
```

    ## 
    ## Permutation test for no association
    ## 
    ## r = 0.716731, p-value = 0.001000
    ## R = 1000 random permuations
    ## Alternative hypothesis: true maximum correlation is not equal to 0

# 9. Spearman permutation test results

The Spearman permutation results are very similar to the Pearson
permutation results. This is because permTest() performs canonical
correlation analysis CCA, not simple correlation. CCA finds the
strongest possible linear association between the variable sets, and
this optimization step makes the result insensitive to whether Pearson
or Spearman correlations were used. Therefore in this case both methods
give identical permutation results, although methodologically are
different.

``` r
# subset group 1
g1 <- subset(Diabetes, group == "Normal")

X1 <- as.matrix(g1$relwt)             # first set
Y1 <- as.matrix(g1[, gluc.vars])      # second set

set.seed(123)
perm_normal <- permTest(X1, Y1, R = 1000, method="spearman")
perm_normal
```

    ## 
    ## Permutation test for no association
    ## 
    ## r = 0.545606, p-value = 0.000000
    ## R = 1000 random permuations
    ## Alternative hypothesis: true maximum correlation is not equal to 0

``` r
# subset group 2
g2 <- subset(Diabetes, group == "Chemical_Diabetic")
X2 <- as.matrix(g2$relwt)
Y2 <- as.matrix(g2[, gluc.vars])

set.seed(123)
perm_chemical <- permTest(X2, Y2, R = 1000, method="spearman")
perm_chemical
```

    ## 
    ## Permutation test for no association
    ## 
    ## r = 0.581793, p-value = 0.030000
    ## R = 1000 random permuations
    ## Alternative hypothesis: true maximum correlation is not equal to 0

``` r
# subset group 3
g3 <- subset(Diabetes, group == "Overt_Diabetic")
X3 <- as.matrix(g3$relwt)
Y3 <- as.matrix(g3[, gluc.vars])

set.seed(123)
perm_overt <- permTest(X3, Y3, R = 1000, method="spearman")
perm_overt
```

    ## 
    ## Permutation test for no association
    ## 
    ## r = 0.758679, p-value = 0.000000
    ## R = 1000 random permuations
    ## Alternative hypothesis: true maximum correlation is not equal to 0
