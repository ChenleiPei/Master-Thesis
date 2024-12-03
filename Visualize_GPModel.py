import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gpytorch

# 定义 GPModel 类
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale, outputscale, noise):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)  # set the lengthscale
        self.covar_module.outputscale = torch.tensor(outputscale)  # set the outputscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 加载模型
model_path = "GPmodel_state_dict.pth"  # 模型路径

# 假设 train_x 和 train_y 是训练数据
dimension = 2
train_x = torch.randn(100, dimension)  # 示例输入数据，维度为 (100, 2)
train_y = torch.randn(100)  # 示例标签
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model = GPModel(train_x, train_y, likelihood, lengthscale=1.0, outputscale=1.0, noise=1e-3)

# 加载模型的 state_dict
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置为评估模式

# 创建较小网格数据进行预测，减少内存需求
x = np.linspace(-5, 5, 30)
y = np.linspace(-5, 5, 30)
X, Y = np.meshgrid(x, y)

# 将网格数据展平并转化为 PyTorch 张量，确保其形状为 (M, 2)
inputs = np.vstack([X.flatten(), Y.flatten()]).T
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

# 分批预测以减少内存需求
batch_size = 100  # 根据内存情况选择合适的批量大小
mean_list = []

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for i in range(0, len(inputs_tensor), batch_size):
        batch_inputs = inputs_tensor[i:i+batch_size]
        pred = model(batch_inputs)
        mean_list.append(pred.mean.numpy())

# 将所有批次的预测结果合并
mean = np.concatenate(mean_list)

# 将均值转换为三维图形所需的形状
Z = mean.reshape(X.shape)

# 创建 3D 图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 图形，X 和 Y 是样本点位置，Z 是预测的均值
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Mean Prediction')

# 添加标题
ax.set_title('3D Visualization of GP Model Mean Predictions')

# 显示图形
plt.show()

