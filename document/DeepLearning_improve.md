# Week1

## 1.Train/dev/test sets

1. Traing set
2. Dev set (validation set)
3. Test set(final test)

In the old era, the percentage is usually 60%/20%/20%

In the big data era, maybe 99.5%/0.25%/0.25%

It might be okay to only have traing set and dev set.


## 2.Bias/Variance
high bias : underfitting

high variance : overfitting

### How to tell high bias or high variance ?

By checking the traning set and dev set error rate.

Traing 1% Dev 11%  => High variance

Traing 15% Dev 16% => High bias

Traing 0.5% Dev 1% => Low bias, low variance

Traning 15% Dev 30% => High bias, high variance



**All this is based on the assumption that Bayes error is quite small**

### How does high bias meanwhile high variance looks like
example a 2D image

## 3. Basic recipe for machine learning

1. Check if it has high bias
- pick other/bigger network
- Train longer(try some advanced algorithm)

2. Once having low bias, check if it has high variance
If it has, then
- More data
- Regularization


## 4. Regularization
Cost function regularization

### 1. Logistic regression cost function regularization (L2 regularization)

$$J(w,b) = 1/m \sum_{i=1}^m L(y^{\^{}(i)},y^{(i)}) + \lambda/{2m} * \| \mathbf{w} \|_2^2 $$ 

$$ \| \mathbf{w} \|_2^2 =  \sum_{j=1}^{n^x} w_j^2 = w^T * w$$

**Note**

- subscript 2 stands for L2 regularization.
- w is a [$n^x$, 1] matrix
- input x is a [$n^x$,1] matrix

### 2. Neural network cost function regularization (Frobenius regularization)

$$J(w,b) = 1/m \sum_{i=1}^m L(y^{\^{}(i)},y^{(i)}) + \lambda/{2m} * \sum_{l=1}^l\| \mathbf{w^{[l]}} \|_2^2 $$

$$ \| \mathbf{w^{[l]}} \|_2^2 = \sum_{i=1}^{n^{[l]}} \sum_{j=1}^{n^{[l-1]}} w_{ij}^2$$

It means square all the weight $w_{ij}$ and sum all of them

**Note**

$w^{[l]}$ is a ($n^{[l]}, n^{[l-1]}$) matrix   

x is a ($n^0$,1) matrix

X is a ($n^0$, m) matrix



### How it impacts gradient descent
Previously

$dw^{[l]}$ = {from backprop}

Now 

$dw^{[l]}$ = {from backprop} + $\lambda/m * w^{[l]}$

$w^{[l]}$ = $w^{[l]} - \alpha * dw^{[l]}$ = $w^{[l]} - \alpha * \lambda/m - \alpha * {from backprop}$ 

It's called weight decay

## 5. Why regularization reduce overfitting
One intuition is if $\lambda$ is very large, $w^{[l]}$ will be close to 0. Then many hidden units will be zeroing out. Eventually we will have a simpler network (like logistic)which lead to less overfitting.

Another explanation is $\lambda$ is large, $w^{[l]}$ will be small. 
And because $z^{[l]}$ = $w^{w[l]} * a^{[l]}$, then $z^{[l]}$ will be small.
If the activation function is tanh and  $z^{[l]}$ is small, the activation function will be close to linear. If every layer has linear activation function, the whole network will be linear. thus reducing overfitting.

## 6. Dropout regularization
A powerful regularization techniques

randomly dropout nodes in neural network

Inverted dropout
``` 
a3 = np.multiply(a3, d3)
a3 /= keep_prob
```

Train each example with a new diminished network.
On iteration one of gradient descent, zero out some hidden units
On iteration two of gradient descent, zero out some hidden units

Not use dropout at test time
### How to implement it ? 

Mostly common way is to use inverted dropout


## 7. Understanding dropout
Why does dropout work.

Intuition:
Can't rely on any one feature, so have to spread out weights

Commonly used in Computer Vision as always lack of data so tend to be overfitting.

You can set keep_prob differently for different layers.
Or zero out only for some layers.

When debug, one can first turn off dropout to see if gradient descent works well.

## 8. Other regulation methods
1. Data augmentation

2. Early stopping
Plot traing set error(or J) and dev set error

**downside**
Deep learning usually takes two steps
- Optimize J
- Not overfit (redudce variance)

Normally it is good or easier to seperate these two tasks. (Called orthogonalization)

But early stopping decouples these two tasks.

**Good side**
If using L2 regularization one need to try many different $\lambda$.
With early stopping one just need to try several w.

L2 regulation is more recommended.

## 9. Normalizing inputs
Normalizing is to speed the up the training.
Two steps
1. Substract the mean

$u=1/m * \Sigma_1^m x^{(i)}$

$x: = x - u$

2. Normalize the variance

$\sigma^2 = 1/m * \Sigma_1^m (x^{(i)})^2$  (x already substract the mean)

$x = x/\sigma$

Use the same $u$ for traing set and test set

Why normalize speeds up ?

![normalize](normalize.png)

## 10. Vanishing/Exploding gradients
For a deep neural network, the output can be quite large or quite small.

For example a deep neural network with $w^{[l]}$ as the identity matrix ((1.5,0),(0,1.5))

![](vanishing_exploding.png)

Similarly the derivative or gradient can also be quite large or small which makes traing difficult.

One partial solution to overcome this is to carefully choose the initial weights.

## 11. Weight initialization for deep nerual network
For a single neuron, ideally we wish the larger the n (number of input feature), the smaller w[i] to be.

variance(w) = 2/n ( it is not 1/n because 2/n works better, the idea is the same)

np.random.rand(shape) * sqaure root(2/n)

Other variance.

**Implementation**
``` python
for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*math.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))*math.sqrt(2./layers_dims[l-1])
```


## 12.Numerical approxiamation of gradients
#### Gradient checking 

check if back-prop is correct

say $f(\theta) = \theta^3$

$\lim_{\epsilon \to 0} (f(\theta+\epsilon) - f(\theta-\epsilon))/2\epsilon = O(\epsilon^2) $ (This is more accurate than below one)



$\lim_{\epsilon \to 0} (f(\theta+\epsilon) - f(\theta))/\epsilon = O(\epsilon) $


## 13. Gradient checking
To help debug and verify implementation is correct.

Take $W^{[1]} b^{[1]} ... W^{[l]} b^{[l]}$ and reshape into a big vector $\theta$

Take $dW^{[1]} db^{[1]} ... dW^{[l]} db^{[l]}$ and reshape into a big vector $d\theta$

Now the question is:

Is $d\theta$ the gradient/slope of cost function $J(\theta)$

#### How to implement it
$J(\theta)$ can be represented as $J(\theta1, \theta2, ..)$

for each i, compute $d_{approx}[i]$

$J(\theta_{approx}[i])$ = $J(\theta1, \theta2, ... \theta(i)+\epsilon ,..)$ - $J(\theta1, \theta2, ... \theta(i)-\epsilon ,..)$ / 2 $\epsilon$

This is appoximate to $d\theta[i]$

Now compare these two vectors with same dimension, are they appoximate to each other.

How ?

compute $$ \| \mathbf{d{\theta_{approx}} - d\theta} \|_2  /  (\| \mathbf{d{\theta_{approx}}} \|_2 + \| \mathbf{d{\theta}} \|_2)$$

The smaller the value, the more likely the implementation is correct.

## 14. Gradient checking implementation notes
- Don't use for traing, only for debug. (It takes extra time)
- If algorithm fails grad check, look at components to try to identity bug
- Remember regularization
- Doesn't work with dropout
- Run at random initialization, perhaps again after some training (because in rare case, your algorithm is correct only when w and b is close to zero, when they grow bigger algorithm doesn't work correctly)


# Week 2

## 1. Mini-batch gradient descent
Split up your training data set for both X and Y
If you have 5 million data, split up to 1000 mini-batch each.

for t 1 ... 5000:
        forward_propagation(x{t}, y{t})
        compute_cost_function()
        backward_propagation()
        $w^{[l]} = w^{[l]} - \alpha * dw^{[l]}$

epoch is one single pass of traing set
Runs faster than batch gradient descent if large data set
## 2. Understanding mini-batc gradient descent
For mini-batch, the cost function should trend down but with some noisy.
It does not always go down after an epoch.

### Choosing your mini-batch size
If m = number of dataset, it takes too long per iteration

If m = 1, it loses the speed up from vectorization

**Guidelines**
If has as small traing set , use batch gradient descent (say m <= 2000)
Typical mini-batch size, 64, 128, 256, 512 (better be a power fo 2)
Fits in CPU/GPU memory.

## 3. Exponetially weighted averages
$V_t = \beta * V_{t-1} + (1-\beta) * \theta_t$

$V_t$ are approximately average over $1/(1-\beta)$ days
![](images/exponentially_weighted_average.png)
## 4. Understanding exponentially weighted averages

### Understanding the formula

### Implementation

## 5. Bias correction in exponentially weighted averages
when t is small, $v_t$ starts with relatively small value.
To gain a better estimation, we can use bias correction.

$v_t/(1-\beta^t)$


## 6. Gradient descent with momentum

smooth out the steps of gradient descent.


## 7. RMSProp

## 8. Adam Optimization Algorithm