# Deep Learning
https://course.fast.ai/Lessons/lesson1.html


## What is a Neural Network

## Supervised Learning with Neural Networks


- Standard NN
- CNN
- RNN

Structured data 
- database

Unstructured data
- audio
- images
- text


## Why is Deep Learning Suddenly Work So Well 


## Binary Classification

X is a $n_x * m$ dimension matrix

Y is a $1 * m$ dimension matrix

in python Y.shape = (1, m) matrix

## Logistic Regression
use sigmoid function to output value from 0 to 1.

## Logistic Regression Cost Function
![cost](images/cost.png)

## Derivatives

## More Derivative Examples

## Computation Graph

## Deravative with a Computation Graph
backpropagation

Chain rule

![computation graph](images/compuation_graph.png)

$dj/dv$=3

$dj/da$=$dj/dv$ * $dv/da$ = 3

$dj/db$=$dj/dv$ * $dv/du$ * $du/db$ = 3 * 1 * 2 = 6

$dj/dc$=$dj/dv$ * $dv/du$ * $du/dc$ = 3 * 1 * 3 = 9

## Logistic Regression Gradient Descent
![gradient](images/logstic_reg_gradient.png)

$z=w^T*X + b$

> say it has 5 features, then W's shape is [5,1], X [5,1]. b and z are numbers.

> so we have [1] = [1,5] * [5,1] + [1]

$a=sigmoid(z)$



## Gradient descent on M examples
![gradient_m_example](images/gradient_m_examples.png)

pesudo code to calculate parital derative dw1, dw2 and db

$$dw1 = 1/m\sum_{i=1}^{n}dw_i $$

to speed up use Vectorization


### Algorithm
![alg](images/gradient_descent_multiple.png)

## Vectorization

Both CPU and GPU support SIMD (Single instruction multiple data)


## More Vectorization Examples
whenever possible, avoid using for loops
matrix * vector
np.exp(v)

eliminate second for loop in gradient descent of logistic regression 


## Vectorizing Logistic Regression


## Vectorizing Logistic Regression's Gradient Computation
![vectorizing](images/vectorizing_logistic_regression_gradient.png)

say we have n features and m training examples.
$ A = sigmoid(w^T * X + b) $
[1,m] = [1,n] * [n,m]



## BroadCasting in Python
![broadcasting](images/broadcasting.png)

## A Note on Python/Numpy Vectors
Not use rank 1 array like (5,), but use (5,1)
Use assert to check np array shape


## Explanation of Logistic Regression Cost Function
![explanation](images/explain_LR_cost.png)

# One Hidden Layer Neural Network (Week 3)

## 3.1 Neural Network Overview

## 3.2 Neural Network Representation
Input layer
Hidden layer
Output layer

In convention usage, the number of layers does not include input layer.


## 3.3 Computing a Neural Network Output
For single training example, say we have 3 feature $x_1$ , $x_2$ and $x_3$. And we have 4 node in hidden layer. Here is how we can compute the output.

$z^{[1]}_{1}$ = $w^{[1]T}_{1} * x + b^{[1]}_{1}$  ([1] means layer 1, subscript 1 means node 1 in this layer)

> (1,1) = (1,3) * (3,1) + (1,1)

$a^{[1]}_{1} = sigmoid(z^{[1]}_{1})$

...

$z^{[1]}_{4}$ = $w^{[1]T}_{4} * x + b^{[1]}_{4}$ 

$a^{[1]}_{4} = sigmoid(z^{[1]}_{4})$

We can vectorize the compute instead of using for loop for above computation.

$z^{[1]} = W^{[1]} * x + b^{[1]}$

> (4,1) = (4,3) * (3,1) + (4,1)

$a^{[1]} = sigmoid(z^{[1]})$

> note $x=a^{[0]}$
. ANd W here is a vector, each row is $w^{[i]T}_j$ , which is transposed.
---
$z^{[2]}$ = $w^{[2]} * a^{[1]}  + b^{[2]}$ 

$a^{[2]} = sigmoid(z^{[2]})$

## 3.4 Vectorizing Across Multiple Training Examples

## 3.5 Explanation of Vectorized Implementation
![vectorize](images/vectorize.png)

For $Z^{[1]}$,(n,m) = (n,nx) * (nx,m) + (n,1) where n is the number of node in layer 1. m is number of training examples. $n_x$ is feature number.
## 3.6 Activation Functions
![activations](images/activations.png)

use sigmoid only when it is a binary classification where the output is 0 or 1.

tanh is strictly superior than sigmoid

ReLu is widely used.

## 3.7 Why Do you Need Non-linear Activation Functions
If you were to use linear/identity activation functions , neural network is just outputing linear function of the input. No matter how many layers you have.


## 3.8 Derivatives of Activation Functions
sigmoid

$g^,(z) = g(z)*(1-g(z)) = a*(1-a)$

tanh

$g^,(z) = 1 - (tanh(z))^2$

ReLu

$g^,(z) = 0$  when z<0

$g^,(z) = 1$  when z>=0

Leaky ReLu

$g^,(z) = 0.01$  when z<0

$g^,(z) = 1$  when z>=0


## 3.9 Gradient Descent for Neural Network
![derivatives](images/deepLearning_derivatives.png)



## 3.10 Backpropagation Intuition


![summary](images/summary.png)


$dw^{[2]} = dz^{[2]} * a^{[1]T}$

$(n^{[2]},n^{[1]}) = (n^{[2]},1) * (1, n^{[1]}) $


## 3.11 Random Initialization


Note: 
w is column vector, (n,1)
x is also column vector, (n,1) ?





## 4.1 Deep L-layers Neural Networks
![deep neural network](images/deep_neural_network.png)

Notations

## 4.2 Forward Propagation in a Deep Network
![forward propa](images/forward_propa.png)


## 4.3 Getting Your Matrix Dimension Right


## 4.4 Why Deep Representations
![intuition1](images/why_deep_nn_works.png)

![intuition2](images/why_deep_nn_works2.png)

## 4.5 Building Blocks of Neural Networks
![forward_backward](images/forward_backward.png)

this is single loop of gradient descent

why computing $da^{[l]}$ ?

## 4.6 Foward and Backward Propagation
![layer L](images/propagation_layer_l.png)

![summary](images/summary.png)


$\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } = \frac{1}{m} (a^{[2](i)} - y^{(i)})$

$\frac{\partial \mathcal{J} }{ \partial W_2 } = \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } a^{[1] (i) T} $

$\frac{\partial \mathcal{J} }{ \partial b_2 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)}}}$

$\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} } =  W_2^T \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } * ( 1 - a^{[1] (i) 2}) $

$\frac{\partial \mathcal{J} }{ \partial W_1 } = \frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} }  X^T $

$\frac{\partial \mathcal{J} _i }{ \partial b_1 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)}}}$

- Note that $*$ denotes elementwise multiplication.
- The notation you will use is common in deep learning coding:
    - dW1 = $\frac{\partial \mathcal{J} }{ \partial W_1 }$
    - db1 = $\frac{\partial \mathcal{J} }{ \partial b_1 }$
    - dW2 = $\frac{\partial \mathcal{J} }{ \partial W_2 }$
    - db2 = $\frac{\partial \mathcal{J} }{ \partial b_2 }$
    


 ## 4.7 Parameters and Hyperparameters
![hyper](images/hyper_parameter.png)

- learning_rate
- number of iterations
- number of layers
- hidden units in different layers
- choice of activation functions

 ## What does have to do with the brain




 # Python basics

 ## np.array
create two dimensional array whose shape is [1,3]

 `np.array([[1,2,3]]) `

 Two dimensional array whose shape is [3,1]

 `np.array([[3],[2],[1]])`


## np.sum
 np.sum 是 numpy（Python中的一个科学计算库）中的函数，用于数组中元素的求和。
该函数的基本语法如下：
np.sum(a, axis=None, dtype=None, out=None, keepdims=False)
参数解释：

a：输入数据，可以是具有数字类型的 list 或 array。
axis：可选参数，表示需要求和的轴，如果没有特别指定，则对所有元素求和。

如果 axis 为 0，那么就会对列求和。
如果 axis 为 1，那么就会对行求和。


dtype：可选参数，如果指定，它必须是数据的预期类型，否则将应用默认规则进行类型转换后再运算求和。
out：可选参数，如果指定，结果将被插入到此参数中，而非产生新的数组。
keepdims：可选参数，确定输出数组是否应保持原来数组的维度，如果为 True，则原来数组的维度被保持在结果中，例如一维数组求和后还是一维数组。

例如：求一个 2x2 的二维数组的列的和：

## np.squeeze

## np.mean

## np.abs

## print
`print("hello {}".format("james"))`

> why np.sum, why w.reshape

