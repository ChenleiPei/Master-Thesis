import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gpytorch
import matplotlib.cm as cm

# 设置随机数种子，以便可重复的结果
torch.manual_seed(42)
np.random.seed(42)

# 定义 GPModel 类
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood):
        # 这里用 None 初始化以跳过训练过程
        super(GPModel, self).__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 加载模型
model_path = "GPmodel_state_dict.pth"  # 模型路径

# 初始化 GaussianLikelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# 创建 GPModel 实例
model = GPModel(likelihood)

# 加载模型的 state_dict
model.load_state_dict(torch.load(model_path))
model.eval()
likelihood.eval()

# 检查模型的某些参数
print("Mean module constant:", model.mean_module.constant.item())
print("Lengthscale:", model.covar_module.base_kernel.lengthscale.item())
print("Outputscale:", model.covar_module.outputscale.item())


# 创建较小网格数据进行预测，减少内存需求
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# 将网格数据展平并转化为 PyTorch 张量，确保其形状为 (M, 2)
inputs = np.vstack([X.flatten(), Y.flatten()]).T
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

# 分批预测以减少内存需求
batch_size = 50  # 进一步减少批量大小
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

# 加载 accumulated samples 和对应的数学表达式
with open('accumulated_samples.json', 'r') as f:
    accumulated_samples_list = json.load(f)

with open('math_expressions.json', 'r') as f:
    math_expressions_list = json.load(f)

# 确保每个 accumulated_samples 中的点是二维的
accumulated_samples_list = [sample for sample in accumulated_samples_list if len(sample) == 2]

# 创建颜色映射
unique_expressions = list(set(math_expressions_list))  # 获取唯一的数学表达式
colors = cm.get_cmap('tab10', len(unique_expressions))  # 使用 tab10 调色板创建颜色映射

# 创建表达式到颜色索引的映射
expression_to_color = {exp: colors(i) for i, exp in enumerate(unique_expressions)}

# 创建 3D 图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 曲面图，X 和 Y 是样本点位置，Z 是预测的均值
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# 绘制 accumulated_samples 并使用相同表达式的点相同颜色，设置在 Z 最低位置
z_bottom = Z.min()  # 获取 Z 平面的最小值，作为底部 z 位置
for sample, expression in zip(accumulated_samples_list, math_expressions_list):
    if len(sample) == 2:  # 确保样本是二维点
        point = np.array(sample)
        color = expression_to_color[expression]
        ax.scatter(point[0], point[1], z_bottom, color=color, s=50, alpha=1.0, edgecolors='k')  # 增大点的大小

# 调整视角，使底部的点可见
ax.view_init(elev=30, azim=120)

# 添加图例
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=expression_to_color[exp], markersize=10)
           for exp in unique_expressions]
ax.legend(handles, unique_expressions, loc='upper right', bbox_to_anchor=(1.3, 1))

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Mean Prediction')

# 添加标题
ax.set_title('3D Visualization of GP Model Mean Predictions with Accumulated Samples at Bottom')

# 显示图形
plt.show()
