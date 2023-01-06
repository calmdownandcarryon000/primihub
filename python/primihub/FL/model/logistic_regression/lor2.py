"""
描述: aby3中LR明文实现
"""
import pandas as pd
import numpy as np


dataset_path = '/home/chainmaker/wz/myPrimihub/primihub/data/DataFile_party.csv'

dataset = np.genfromtxt(dataset_path, delimiter=',')
#数据集前面加一列全为1
dataset = np.hstack([np.ones((dataset.shape[0], 1)), dataset])  
dim = dataset.shape[0]
dim_train = (int)(dim / 10 * 8)

# train_data = dataset.loc[:dim_train, :].reset_index(drop=True)
# test_data = dataset.loc[dim_train:dim, :].reset_index(drop=True)
train_data = dataset[:dim_train, :]
test_data = dataset[dim_train:dim, :]

y = train_data[:, -1]
x = train_data[:, 0:-1]
test_y = test_data[:, -1]
test_x = test_data[:, 0:-1]
# 初始化权重值
w = np.zeros(x.shape[1])  
iters = 2000
rate = 1.0 / (1 << 6);
print(x)
print(w)
print(x.shape)
print(w.shape)
print(np.dot(x,w))
print(y)
#Train
for i in range(1):
    #假设函数 sigmoid
    predict = np.dot(x,w)


    h_x = 1/(1+np.exp(-predict))
    print(h_x)
    #损失函数
    error = h_x - y
    
    loss = (1/dataset.shape[0])*(x.T.dot(error))
    print(1/dataset.shape[0])
    print(x.T.shape)
    print(x.T)
    print(error.shape)
    print(error)

    w = w - rate*loss
    
#Test
out = 1/(1+np.exp(-np.dot(test_x,w)))
count = 0;
for i in range(out.size):
    c0 = out[i] > 0.5;
    c1 = test_y[i] > 0.5;

    count += (c0 == c1);

print(count/test_y.size)



# model = LogisticRegression()
# model.fit(x, y)




#     def main():
#         # 读取训练集(txt文件)中的数据,
#         data, labels = loadDataset()
#         # 将数据转换成矩阵的形式，便于后面进行计算
#         # 构建特征矩阵X
#         X = np.array(data)
#         # 构建标签矩阵y
#         y = np.array(labels).reshape(-1,1)
#         # 随机生成一个w参数(权重)矩阵    .reshape((-1,1))的作用是，不知道有多少行，只想变成一列
#         W = 0.001*np.random.randn(3,1).reshape((-1,1))   
#         # m表示一共有多少组训练数据
#         m = len(X)
#         # 定义梯度下降的学习率 0.03
#         learn_rate = 0.03

#         loss_list = []
#         # 实现梯度下降算法，不断更新W,获得最优解，使损失函数的损失值最小
#         for i in range(3000):
#             # 最重要的就是这里用numpy 矩阵计算，完成假设函数计算，损失函数计算，梯度下降计算
#             # 计算假设函数 h(w)x
#             g_x = np.dot(X,W)
#             h_x = 1/(1+np.exp(-g_x))

#             # 计算损失函数 Cost Function 的损失值loss
#             loss = np.log(h_x)*y+(1-y)*np.log(1-h_x)
#             loss = -np.sum(loss)/m
#             loss_list.append(loss)

#             # 梯度下降函数更新W权重
#             dW = X.T.dot(h_x-y)/m
#             W += -learn_rate*dW
#         test_x = np.array([1,-1.395634,4.662541])
#         test_y = 1/(1+np.exp(-np.dot(test_x,W)))
#         print(test_y)

# class LogisticRegression:
#     def __init__(self):
#         """
#         初始化 Logistic Regression 模型
#         """
#         self.coef = None  # 权重矩阵
#         self.intercept = None  # 截距
#         self._theta = None  # _theta[0]是intercept,_theta[1:]是coef

#     def sigmoid(self, x):
#         """
#         sigmoid 函数
#         :param x: 参数x
#         :return: 取值y
#         """
#         y = 1.0 / (1.0 + np.exp(-x))
#         return y

#     def loss_func(self, theta, x_b, y):
#         """
#         损失函数
#         :param theta: 当前的权重和截距
#         :param x_b: 修改过的数据集（第一列为全1）
#         :param y: 数据集标签
#         :return:
#         """
#         p_predict = self.sigmoid(x_b.dot(theta))
#         try:
#             return -np.sum(y * np.log(p_predict) + (1 - y) * np.log(1 - p_predict))
#         except:
#             return float('inf')

#     def d_loss_func(self, theta, x_b, y):
#         """
#         损失函数的导数
#         :param theta: 当前的权重和截距
#         :param x_b: 修改过的数据集（第一列为全1）
#         :param y: 数据集标签
#         :return:
#         """
#         out = self.sigmoid(x_b.dot(theta))  # 计算sigmoid函数的输出结果
#         return x_b.T.dot(out - y) / len(x_b)

#     def gradient_descent(self, x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
#         """
#         梯度下降函数
#         :param x_b: 修改过的数据集（第一列为全1）
#         :param y: 数据集标签
#         :param initial_theta: 初始权重矩阵
#         :param eta: 学习率
#         :param n_iters: 最大迭代周期
#         :param epsilon: 当两次训练损失函数下降小于此值是提前结束训练
#         :return:
#         """
#         theta = initial_theta
#         i_iter = 0
#         while i_iter < n_iters:
#             gradient = self.d_loss_func(theta, x_b, y)
#             last_theta = theta
#             theta = theta - eta * gradient
#             i_iter += 1
#             if abs(self.loss_func(theta, x_b, y) - self.loss_func(last_theta, x_b, y)) < epsilon:
#                 break
#         return theta

#     def fit(self, train_data, train_label, eta=0.01, n_iters=1e4):
#         """
#         模型训练函数
#         :param train_data: 训练数据集
#         :param train_label: 训练数据标签
#         :param eta: 学习率，默认为0.01
#         :param n_iters: 最大迭代次数
#         :return:
#         """
#         assert train_data.shape[0] == train_label.shape[0], "训练数据集的长度需要和标签长度保持一致"
#         x_b = np.hstack([np.ones((train_data.shape[0], 1)), train_data])  # 在原有数据集前加全1列
#         initial_theta = np.zeros(x_b.shape[1])  # 初始化权重值
#         self._theta = self.gradient_descent(x_b, train_label, initial_theta, eta, n_iters)  # 使用梯度下降训练数据
#         self.intercept = self._theta[0]  # 得到截距
#         self.coef = self._theta[1:]  # 得到权重矩阵
#         return self

#     def predict_proba(self, x_predict):
#         """
#         得到预测的实际结果
#         :param x_predict: 待预测值
#         :return: 预测的实际结果
#         """
#         x_b = np.hstack([np.ones((len(x_predict)), 1), x_predict])
#         return self.sigmoid(x_b.dot(self._theta))

#     def predict(self, x_predict):
#         """
#         对数据进行分类
#         :param x_predict: 待分类的数据集
#         :return: 数据集分类
#         """
#         proba = self.predict_proba(x_predict)
#         # 由于sigmoid函数的输出值是在0-1的，因此将小于0.5的归为一类，将大于0.5的归为一类
#         return np.array(proba > 0.5, dtype='int')

