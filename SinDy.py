import numpy as np
import pysindy as ps
import wandb

wandb.login(key="1a52f6079ddb0d4c0e9f5869d9cc0bdd3f5d9a01")
wandb.init(project='SInDy_analysis', name='Koza_3')

data_train = np.load('Datasets/Koza_3_train.npy')
data_test = np.load('Datasets/Koza_3_test.npy')

print(data_train)

# 准备特征和目标变量
X_train = data_train[:, 0].reshape(-1, 1)
y_train = data_train[:, 1]
X_test = data_test[:, 0].reshape(-1, 1)
y_test = data_test[:, 1]

# 初始化结果存储列表
rmse_train_list = []
rmse_test_list = []

# 运行模型训练和测试 10 次
for _ in range(10):
    # 初始化并拟合 SINDy 模型
    model = ps.SINDy()
    model.fit(X_train, x_dot=y_train, quiet=True)

    # 使用训练数据进行预测
    y_pred_train = model.predict(X_train)
    rmse_train = np.sqrt(np.mean((y_pred_train - y_train) ** 2))
    rmse_train_list.append(rmse_train)

    # 使用测试数据进行预测
    y_pred_test = model.predict(X_test)
    rmse_test = np.sqrt(np.mean((y_pred_test - y_test) ** 2))
    rmse_test_list.append(rmse_test)

    print(f"RMSE Train: {rmse_train}, RMSE Test: {rmse_test}")

# 计算均值和标准差
mean_rmse_train = np.mean(rmse_train_list)
std_rmse_train = np.std(rmse_train_list)

mean_rmse_test = np.mean(rmse_test_list)
std_rmse_test = np.std(rmse_test_list)
wandb.log({
    'Mean RMSE Train': mean_rmse_train,
    'Std RMSE Train': std_rmse_train,
    'Mean RMSE Test': mean_rmse_test,
    'Std RMSE Test': std_rmse_test
})
# 打印结果
print(f"Mean RMSE Train: {mean_rmse_train}, Standard Deviation: {std_rmse_train}")
print(f"Mean RMSE Test: {mean_rmse_test}, Standard Deviation: {std_rmse_test}")


