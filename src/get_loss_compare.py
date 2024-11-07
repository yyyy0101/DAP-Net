import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def read_loss_from_file(file_path, num_epochs=None):
    with open(file_path, 'r') as file:
        # 读取每一行的损失值，并转换为浮点数
        loss_values = [float(line.strip()) for line in file.readlines()]
    return loss_values[:num_epochs]

# 假设两个模型的损失值分别保存在 model1_loss.txt 和 model2_loss.txt 中
model1_loss_path = 'logs/loss_2023_11_14_17_24_22/epoch_loss.txt'
model2_loss_path = 'logs/loss_2024_01_09_21_53_00/epoch_loss.txt'

num_epochs_to_compare = 300

# 读取损失值
model1_losses = read_loss_from_file(model1_loss_path,num_epochs_to_compare)
model2_losses = read_loss_from_file(model2_loss_path,num_epochs_to_compare)



# 训练轮次（假设轮次数相同）
epochs = range(1, len(model1_losses) + 1)

# 画图
line1, = plt.plot(epochs, model1_losses, color='red',linestyle="solid",label='CrossEntropy Loss')
line2, = plt.plot(epochs, model2_losses, color='blue',linestyle="solid", label='Improved Focal Loss')

triangle_marker_positions = [0,25,50,75,100,125,150,175,200,225,250,275,299]
plt.scatter(triangle_marker_positions, [model1_losses[i] for i in triangle_marker_positions], marker='^', color='red')
plt.scatter(triangle_marker_positions, [model2_losses[i] for i in triangle_marker_positions], marker='s', color='blue')


legend_elements = [
    Line2D([0], [0], color='red', linestyle="solid", marker='^', markersize=6, label='CrossEntropy Loss'),
    Line2D([0], [0], color='blue', linestyle="solid", marker='s', markersize=6, label='Improved Focal Loss'),
]





# 设置图形标题和轴标签
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 添加图例
plt.legend(handles=legend_elements)

# 显示图形
plt.show()
