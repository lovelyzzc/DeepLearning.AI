# conda install scipy==1.2.1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import scipy.misc
from lr_utils import load_dataset


# -------------第一步：创建dataset（）导入数据---------------------
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
"""
- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- a test set of m_test images labeled as cat or non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). 
Thus, each image is square (height = num_px) and (width = num_px).
"""

# 计算训练集，测试集的大小和图片的大小（长，宽）
# 图片形状可以用shape函数得出
# print(train_set_x_orig.shape)
# (209, 64, 64, 3)
# 图片数据 arry = [m_train, num_px, num_px, 3
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]


# 数据预处理
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T


# 数据归一化处理
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# -------------------第二步：构建函数------------------------
# 构建需要的函数
# sigmoid
def sigmoid(z):
    """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
    """
    s = 1.0 / (1 + np.exp(-z))

    return s


# initialize_with_zeros
def initialize_with_zeros(dim):
    """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        Returns:
        w -- initialized vector of shape (dim, 1)
        b --
    """
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# Forward and Backward propagation
def propagate(w, b, X, Y):
    """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1.0/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    dw = (1.0/m) * np.dot(X, (A - Y).T)
    db = (1.0/m) * np.sum(A-Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {'dw': dw,
            'db': db}

    return grads, cost


# 创建训练器
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and (i + 1) % 100 == 0:
            print('Cost after iteration {} : {}'.format(i + 1, cost))

    params = {'w': w,
              'b': b}

    grads = {'dw': dw,
             'db': db}

    return params, grads, costs


# 预测函数
def predict(w, b, X):
    """
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# --------------------第三步：构建模型------------------------
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
       Builds the logistic regression model by calling the function you've implemented previously

       Arguments:
       X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
       Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
       X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
       Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
       num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
       learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
       print_cost -- Set to true to print the cost every 100 iterations

       Returns:
       d -- dictionary containing information about the model.
    """

    # 初始化参数
    w, b = initialize_with_zeros(X_train.shape[0])

    # 梯度下降训练
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 更新参数
    w = parameters['w']
    b = parameters['b']

    # 进行预测
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印准确率
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {'costs': costs,
         'Y_prediction_test': Y_prediction_test,
         'Y_prediction_train': Y_prediction_train,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations
    }

    return d


# -------------------运行模型---------------------
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.001, print_cost=True)


# 选取一张照片查看情况
index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
plt.show()
print('y = ' + str(test_set_y[0, index]))
print('prediction = ' + classes[int(d['Y_prediction_test'][0, index])].decode('utf-8'))
costs = np.squeeze(d['costs'])


# 做出cost的变化曲线
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# 做出不同学习率下的cost曲线
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is :" + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate=i)
    print('\n' + '------------------------------------------------------' + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]['costs']), label=str(models[str(i)]['learning_rate']))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# 测试从其他地方下载的图片
# 先运行模型再执行下面的步骤
my_image = 'isacatornot.jpg'
fname = 'images/' + my_image
image = np.array(ndimage.imread(fname))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d['w'], d['b'], my_image)

plt.imshow(image)
print('y = ' + str(np.squeeze(my_predicted_image)))
print('prediction = ' + classes[int(np.squeeze(my_predicted_image)), ].decode('utf-8'))
