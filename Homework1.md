
Name: Lu, Di

PID: A53207935

language used: Python 2.7.13 :: Anaconda

# Preparation

Before the real tasks, first download and read in the data.


```python
import os

if os.path.isfile('beer_50000.json'):
    print('Already downloaded')
else:
    !wget http://jmcauley.ucsd.edu/cse258/data/beer/beer_50000.json
```

    --2017-10-13 17:12:51--  http://jmcauley.ucsd.edu/cse258/data/beer/beer_50000.json
    Resolving jmcauley.ucsd.edu... 137.110.160.73
    Connecting to jmcauley.ucsd.edu|137.110.160.73|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 61156124 (58M) [application/json]
    Saving to: ‘beer_50000.json’
    
    beer_50000.json     100%[===================>]  58.32M  17.8MB/s    in 3.3s    
    
    2017-10-13 17:12:55 (17.7 MB/s) - ‘beer_50000.json’ saved [61156124/61156124]
    



```python
print('Reading data......')
with open('beer_50000.json') as f:
    data = [eval(line) for line in f]
print('Done')
```

    Reading data......
    Done


# Tasks — Regression
In the first three questions, we’ll see how ratings vary across different categories of beer. These questions should be completed on the entire dataset.

## Problem 1
How many reviews are there for each style of beer in the dataset (‘beer/style’)? What is the average value of ‘review/taste’ for reviews from each style? (1 mark)


```python
from collections import Counter, defaultdict
import numpy as np
```

Define a simple function to print the output as table


```python
def print_table(datum):
    for style, count in sorted(datum.items(), key=lambda x: -x[1]):
        print '{:<35}\t{}'.format(style, count)
```

To compute the number of reviews for each style of beer in the dataset:


```python
style_count = Counter([d['beer/style'] for d in data])
print_table(style_count)
```

    American Double / Imperial Stout   	5964
    American IPA                       	4113
    American Double / Imperial IPA     	3886
    Scotch Ale / Wee Heavy             	2776
    Russian Imperial Stout             	2695
    American Pale Ale (APA)            	2288
    American Porter                    	2230
    Rauchbier                          	1938
    Rye Beer                           	1798
    Czech Pilsener                     	1501
    Fruit / Vegetable Beer             	1355
    English Pale Ale                   	1324
    Old Ale                            	1052
    Doppelbock                         	873
    American Barleywine                	825
    Euro Pale Lager                    	701
    Extra Special / Strong Bitter (ESB)	667
    American Amber / Red Ale           	665
    Munich Helles Lager                	650
    Belgian Strong Pale Ale            	632
    Hefeweizen                         	618
    American Stout                     	591
    German Pilsener                    	586
    Pumpkin Ale                        	560
    Märzen / Oktoberfest              	557
    Baltic Porter                      	514
    Light Lager                        	503
    English Brown Ale                  	495
    Wheatwine                          	455
    English Porter                     	367
    American Blonde Ale                	357
    Euro Strong Lager                  	329
    American Brown Ale                 	314
    English Bitter                     	267
    Winter Warmer                      	259
    Tripel                             	257
    American Adjunct Lager             	242
    Maibock / Helles Bock              	225
    English India Pale Ale (IPA)       	175
    Belgian Dark Ale                   	175
    American Strong Ale                	166
    Dubbel                             	165
    Altbier                            	165
    English Strong Ale                 	164
    Witbier                            	162
    American Pale Wheat Ale            	154
    Bock                               	148
    Belgian Strong Dark Ale            	146
    Belgian Pale Ale                   	144
    Euro Dark Lager                    	144
    Munich Dunkel Lager                	141
    Saison / Farmhouse Ale             	141
    American Black Ale                 	138
    English Stout                      	136
    English Barleywine                 	133
    Belgian IPA                        	128
    American Pale Lager                	123
    Black & Tan                        	122
    Quadrupel (Quad)                   	119
    Oatmeal Stout                      	102
    Irish Dry Stout                    	101
    American Wild Ale                  	98
    Kölsch                            	94
    American Malt Liquor               	90
    Irish Red Ale                      	83
    Scottish Ale                       	78
    Herbed / Spiced Beer               	73
    Cream Ale                          	69
    Milk / Sweet Stout                 	69
    Scottish Gruit / Ancient Herbed Ale	65
    Dunkelweizen                       	61
    Smoked Beer                        	61
    Foreign / Export Stout             	55
    Schwarzbier                        	53
    American Amber / Red Lager         	42
    Vienna Lager                       	33
    Dortmunder / Export Lager          	31
    Braggot                            	26
    Keller Bier / Zwickel Bier         	23
    English Dark Mild Ale              	21
    English Pale Mild Ale              	21
    American Dark Wheat Ale            	14
    American Double / Imperial Pilsner 	14
    Weizenbock                         	13
    Flanders Oud Bruin                 	13
    Chile Beer                         	11
    California Common / Steam Beer     	11
    Lambic - Unblended                 	10
    Berliner Weissbier                 	10
    Eisbock                            	8
    Low Alcohol Beer                   	7
    Bière de Garde                    	7
    Kristalweizen                      	7
    Lambic - Fruit                     	6
    Flanders Red Ale                   	2


To compute the average value of ‘review/taste’ for reviews from each style


```python
taste_avg = defaultdict(int)
for d in data:
    style, taste = d['beer/style'], d['review/taste']
    taste_avg[style] += taste / style_count[style]
print_table(taste_avg)
```

    American Double / Imperial Stout   	4.47996311201
    English Barleywine                 	4.36090225564
    Russian Imperial Stout             	4.30037105751
    Rye Beer                           	4.21357063404
    Baltic Porter                      	4.21303501946
    American Wild Ale                  	4.1887755102
    Wheatwine                          	4.18681318681
    Old Ale                            	4.09600760456
    Scotch Ale / Wee Heavy             	4.08339337176
    American Porter                    	4.08183856502
    Rauchbier                          	4.06785345717
    American Barleywine                	4.06424242424
    Belgian Strong Pale Ale            	4.05617088608
    American Double / Imperial IPA     	4.03332475553
    American IPA                       	4.00085096037
    Doppelbock                         	3.98281786942
    Munich Helles Lager                	3.95923076923
    Chile Beer                         	3.95454545455
    Belgian IPA                        	3.94921875
    Black & Tan                        	3.94262295082
    Bière de Garde                    	3.92857142857
    Flanders Oud Bruin                 	3.92307692308
    Scottish Gruit / Ancient Herbed Ale	3.90769230769
    American Black Ale                 	3.8731884058
    Keller Bier / Zwickel Bier         	3.86956521739
    American Double / Imperial Pilsner 	3.82142857143
    American Stout                     	3.81979695431
    Braggot                            	3.80769230769
    Pumpkin Ale                        	3.7875
    English Dark Mild Ale              	3.78571428571
    Tripel                             	3.78404669261
    Milk / Sweet Stout                 	3.78260869565
    Munich Dunkel Lager                	3.78014184397
    Oatmeal Stout                      	3.77450980392
    Scottish Ale                       	3.76282051282
    English Strong Ale                 	3.75609756098
    Lambic - Fruit                     	3.75
    Eisbock                            	3.75
    Maibock / Helles Bock              	3.74666666667
    American Brown Ale                 	3.74363057325
    Belgian Pale Ale                   	3.73958333333
    Dubbel                             	3.73636363636
    English Brown Ale                  	3.72828282828
    English Porter                     	3.70708446866
    Euro Dark Lager                    	3.70486111111
    Saison / Farmhouse Ale             	3.70212765957
    Kölsch                            	3.69680851064
    Belgian Strong Dark Ale            	3.69520547945
    American Amber / Red Lager         	3.69047619048
    Extra Special / Strong Bitter (ESB)	3.68515742129
    American Dark Wheat Ale            	3.67857142857
    German Pilsener                    	3.66723549488
    American Pale Ale (APA)            	3.64969405594
    Hefeweizen                         	3.63511326861
    Irish Dry Stout                    	3.62376237624
    Schwarzbier                        	3.62264150943
    Winter Warmer                      	3.62162162162
    Czech Pilsener                     	3.60959360426
    Fruit / Vegetable Beer             	3.60774907749
    English Stout                      	3.59926470588
    Quadrupel (Quad)                   	3.59663865546
    English Pale Mild Ale              	3.59523809524
    Märzen / Oktoberfest              	3.5933572711
    American Strong Ale                	3.56927710843
    Berliner Weissbier                 	3.55
    English Bitter                     	3.53745318352
    Vienna Lager                       	3.5303030303
    Witbier                            	3.52777777778
    American Amber / Red Ale           	3.51353383459
    Dunkelweizen                       	3.49180327869
    English Pale Ale                   	3.48376132931
    English India Pale Ale (IPA)       	3.47142857143
    Herbed / Spiced Beer               	3.44520547945
    Dortmunder / Export Lager          	3.41935483871
    Altbier                            	3.40303030303
    Weizenbock                         	3.38461538462
    Belgian Dark Ale                   	3.34
    American Pale Wheat Ale            	3.33441558442
    California Common / Steam Beer     	3.31818181818
    Lambic - Unblended                 	3.3
    American Blonde Ale                	3.25490196078
    Foreign / Export Stout             	3.25454545455
    Flanders Red Ale                   	3.25
    American Pale Lager                	3.21544715447
    Smoked Beer                        	3.19672131148
    Bock                               	3.18918918919
    Cream Ale                          	3.02898550725
    Irish Red Ale                      	2.98192771084
    Euro Pale Lager                    	2.96291012839
    American Adjunct Lager             	2.94834710744
    Euro Strong Lager                  	2.84802431611
    Kristalweizen                      	2.78571428571
    Low Alcohol Beer                   	2.71428571429
    Light Lager                        	2.39662027833
    American Malt Liquor               	2.25555555556


## Problem 2 
Train a simple predictor with a single binary feature indicating whether a beer is an ‘American IPA’:

$$\text{review/taste}\simeq \theta_0 + \theta_1 \times \text{[beer is an American IPA]}$$

Report the values of $\theta_0$ and $\theta_1$ . Briefly describe your interpretation of these values, i.e., what do $\theta_0$ and $\theta_1$ represent (1 mark)?


```python
def feature(datum):
    return [1] + [int(datum['beer/style'] == 'American IPA')]

X = [feature(d) for d in data]
y = [d['review/taste'] for d in data]
theta,residuals,rank,s = np.linalg.lstsq(X, y)
print('theta_0 = %f, theta_1 = %f' % (theta[0], theta[1]))
```

    theta_0 = 3.915205, theta_1 = 0.085646


Hence we have $$\theta_0=3.915205$$ $$\theta_1=0.085646$$
Here $\theta_0$ represents the bias, i.e. the average review/taste value for beer that is not American IPA; $\theta_1$ represents how the review/taste value of American IPA differs from the average of other styles of beer. Since here the $\theta_1$ is positive, we know that the review/taste value of American IPA is $0.086$ higher than the average of other styles of beer.

## Problem 3
Split the data into two equal fractions – the first half for training, the second half for testing (based on the order they appear in the file). Train the same model as above on the training set only. What is the model’s MSE on the training and on the test set (1 mark)?


```python
size = len(data)
train_X, test_X = X[:size / 2], X[size / 2:]
train_y, test_y = y[:size / 2], y[size / 2:]
theta,residuals,rank,s = np.linalg.lstsq(train_X, train_y)
print('theta_0 = %f, theta_1 = %f' % (theta[0], theta[1]))
```

    theta_0 = 3.904356, theta_1 = 0.056060



```python
def MSE(actual, pred):
    return np.mean((actual - pred) ** 2)

train_mse = MSE(np.dot(train_X, theta), train_y)
test_mse = MSE(np.dot(test_X, theta), test_y)
print('MSE on training set is %f' % train_mse)
print('MSE on test set is %f' % test_mse)
```

    MSE on training set is 0.558107
    MSE on test set is 0.468410


## Problem 4
Extend the model above so that it incorporates binary features for every style of beer with ≥ 50 reviews. Report the values of $\theta$ that you obtain, and the model’s MSE on the training and on the test set (1 mark).


```python
valid_styles = sorted([style for style in style_count 
                       if style_count[style] >= 50])
style_index = {style: i + 1 for i, style in enumerate(valid_styles)}

def feature(datum):
    feat = [1] + [0] * len(valid_styles)
    style = datum['beer/style']
    if style in style_index:
        feat[style_index[style]] = 1
    return feat

X = [feature(d) for d in data]
y = [d['review/taste'] for d in data]
train_X, test_X = X[:size / 2], X[size / 2:]
train_y, test_y = y[:size / 2], y[size / 2:]
theta,residuals,rank,s = np.linalg.lstsq(train_X, train_y)
print('theta =\n %s' % theta)
```

    theta =
     [ 3.60681818 -0.18884943 -0.73802386 -0.11634199  0.45638407  0.19621212
     -0.45410882  0.26761104  0.33596789  0.8416167   0.35359848 -1.00681818
      0.02739474 -0.58598485 -0.17824675  0.3305986   0.20312334 -0.1798951
      0.58195733  0.65508658 -0.27061129  0.3449362   0.16385851  0.12651515
      0.46842454  0.33580477 -0.74318182 -0.65227273 -0.2937747  -0.41285266
      0.13786267  0.25681818  0.76540404 -0.02450111  0.03420746 -0.13106061
     -0.31931818  0.09815076 -0.10223103  0.12193999  0.11433566 -0.92019846
     -0.83277972  0.14466111 -0.35681818  0.11320382 -0.62765152 -0.0058248
     -0.38622995  0.22651515 -0.63806818  0.10049889 -1.23390152 -0.10681818
      0.71136364 -0.50681818 -0.20681818 -0.22045455  0.1527972   0.60472028
      0.26709486  0.39318182  0.44318182  0.69564175  0.3763751   0.20984848
      0.12395105  0.48544372  0.26818182  0.38356643 -0.4664673   0.05895722
      0.01818182  0.01699134 -0.09003966]



```python
train_mse = MSE(np.dot(train_X, theta), train_y)
test_mse = MSE(np.dot(test_X, theta), test_y)
print('MSE on training set is %f' % train_mse)
print('MSE on test set is %f' % test_mse)
```

    MSE on training set is 0.367840
    MSE on test set is 0.433670


# Tasks — Classification
Next we’ll try to train classifiers that are able to predict a beer’s style from the characteristics of its review. Again, split the data so that the first half is used for training and the second half is used for testing as we did for Q3.

## Problem 5
First, let’s train a predictor that estimates whether a beer is an ‘American IPA’ using two features:

$$\text{[‘beer/ABV’, ‘review/taste’].}$$

Train your predictor using an SVM classifier (see the code provided in class) – remember to train on the first half and test on the second half. Use a regularization constant of C = 1000 as in the code stub. What is the accuracy (percentage of correct classifications) of the predictor on the train and test data? (1 mark)


```python
from sklearn import svm
```


```python
def feature(datum):
    return [datum['beer/ABV'], datum['review/taste']]

def label(datum):
    return int(datum['beer/style'] == 'American IPA')

X = [feature(d) for d in data]
y = [label(d) for d in data]
train_X, test_X = X[:size / 2], X[size / 2:]
train_y, test_y = y[:size / 2], y[size / 2:]
```


```python
clf = svm.SVC(C=1000)
clf.fit(train_X, train_y)
```




    SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
def accuracy(pred, y):
    return np.mean(pred == y)
```


```python
train_predictions = clf.predict(train_X)
test_predictions = clf.predict(test_X)
train_acc = accuracy(train_predictions, train_y)
test_acc = accuracy(test_predictions, test_y)
print('train accuracy is %.2f%%' % (train_acc * 100))
print('test accuracy is %.2f%%' % (test_acc * 100))
```

    train accuracy is 92.26%
    test accuracy is 85.63%


## Problem 6
Considering the ‘American IPA’ style, can you come up with a more accurate predictor (e.g. using features from the text, or otherwise)? Write down the feature vector you design, and report its train/test accuracy (1 mark).

\vbox{}\vbox{}

Since we are looking for American IPA, it's reasonable to have a binary feature which takes the value 1 only if "IPA" is mentioned in review text. To distinguish with those words that has 'ipa' inside (e.g. principal), an empty space is added. Together with the features we had in problem 5, we now have 3 features:

$$\text{[‘beer/ABV’, ‘review/taste’, whether ' ipa' is in 'review/text'].}$$

In addition, to expedite the training process, a normalization is done by linearly mapping training samples to [0, 1], i.e.
$$\widehat{x}=\frac{x - \min(x_{\text{train}}[n])}{\max(x_{\text{train}}[n] - \min(x_{\text{train}}[n]))}$$
Another reason for doing this normalization is that if it is not done, the SVM classifier won't converge when $C=100000$.


```python
def normalization(features, training=False):
    features = np.asarray(features)
    if training:
        normalization.mins = np.min(features, axis=0)
        normalization.maxs = np.max(features, axis=0)
    return (features - normalization.mins) /  \
                (normalization.maxs - normalization.mins)
```


```python
def feature(datum):
    feat = [datum['review/taste'],
            datum['beer/ABV'],
            int(' ipa' in datum['review/text'].lower())]
    return feat

X = [feature(d) for d in data]
y = [label(d) for d in data]
train_X = normalization(X[:size / 2], training=True)
test_X  = normalization(X[size / 2:], training=False)
train_y, test_y = y[:size / 2], y[size / 2:]
```


```python
clf = svm.SVC(C=1000)
clf.fit(train_X, train_y)
```




    SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
train_predictions = clf.predict(train_X)
test_predictions = clf.predict(test_X)
train_acc = accuracy(train_predictions, train_y)
test_acc = accuracy(test_predictions, test_y)
print('train accuracy is %.2f%%' % (train_acc * 100))
print('test accuracy is %.2f%%' % (test_acc * 100))
```

    train accuracy is 94.40%
    test accuracy is 95.06%


Clearly, both train and test accuracy are higher.

## Problem 7
What effect does the regularization constant $C$ have on the training/test performance? Report the train/test accuracy of your predictor from the previous question for $C \in \langle 0.1, 10, 1000, 100000\rangle$.


```python
for C in [0.1, 10, 1000, 100000]:
    clf = svm.SVC(C=C)
    clf.fit(train_X, train_y)
    train_predictions = clf.predict(train_X)
    test_predictions = clf.predict(test_X)
    train_acc = accuracy(train_predictions, train_y)
    test_acc = accuracy(test_predictions, test_y)
    print('-' * 50)
    print('C = %s' % str(C))
    print('train accuracy is %.2f%%' % (train_acc * 100))
    print('test accuracy is %.2f%%' % (test_acc * 100))
```

    --------------------------------------------------
    C = 0.1
    train accuracy is 91.36%
    test accuracy is 92.19%
    --------------------------------------------------
    C = 10
    train accuracy is 93.72%
    test accuracy is 95.32%
    --------------------------------------------------
    C = 1000
    train accuracy is 94.40%
    test accuracy is 95.06%
    --------------------------------------------------
    C = 100000
    train accuracy is 94.30%
    test accuracy is 95.24%


Generally speaking, the larger C is, the more penalization would be given to error samples. Hence, larger C tend to cause overfitting, while smaller C may lead to low accuracy. As we can see, when $C=0.1$, the model does not work well and classifies all samples as negative. However, when C is large the model still works well. This may be because the feature used here has pretty similar distribution on training and test set.

## Problem 8
Finally, let’s fit a model (for the problem from Q5) using logistic regression. A code stub has been provided to perform logistic regression using the above model on http://jmcauley.ucsd.edu/cse258/code/homework1.py Code for the log-likelihood has been provided in the code stub (f) but code for the derivative is incomplete (fprime)

Complete the code stub for the derivative (fprime) and provide your solution. What is the log-likelihood of after convergence, and what is the accuracy (on the test set) of the resulting model (1 mark)?

\vbox{}\vbox{}

Here the features of Q6&Q7 are used, since the feature of Q5 does not work well for Logistic regression.


```python
import numpy
import scipy.optimize
import random
from math import exp
from math import log

def inner(x,y):
    return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
    return 1.0 / (1 + exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
    loglikelihood = 0
    for i in range(len(X)):
        logit = inner(X[i], theta)
        loglikelihood -= log(1 + exp(-logit))
        if not y[i]:
            loglikelihood -= logit
    for k in range(len(theta)):
        loglikelihood -= lam * theta[k]*theta[k]
    print "ll =", loglikelihood
    return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
    dl = [0.0]*len(theta)
    for i in range(len(X)):
        # Fill in code for the derivative
        t = sigmoid(inner(X[i], theta)) 
        dl = map(lambda (a, b): a - (t - y[i]) * b, zip(dl, X[i])) 
    dl -= 2 * lam * theta
    # Negate the return value since we're doing gradient *ascent*
    return numpy.array([-x for x in dl])

theta,l,info = scipy.optimize.fmin_l_bfgs_b(
    f, [0]*len(X[0]), fprime, args = (train_X, train_y, 1.0))
print "Final log likelihood =", -l

train_predictions = np.dot(train_X, theta) > 0
test_predictions = np.dot(test_X, theta) > 0
train_acc = accuracy(train_predictions, train_y)
test_acc = accuracy(test_predictions, test_y)
print "Train Accuracy = %.2f%%" % (train_acc * 100)
print "Test Accuracy = %.2f%%" % (test_acc * 100)
```

    ll = -17328.679514
    ll = -11471.2391492
    ll = -8140.67935678
    ll = -7347.77996679
    ll = -6691.23794532
    ll = -6405.47427458
    ll = -6385.28161571
    ll = -6276.16347065
    ll = -6182.08386228
    ll = -5985.58786338
    ll = -5784.74979628
    ll = -5743.72349792
    ll = -5742.43017335
    ll = -5742.34977879
    ll = -5742.3488141
    ll = -5742.34881082
    Final log likelihood = -5742.34881082
    Train Accuracy = 93.91%
    Test Accuracy = 93.70%

