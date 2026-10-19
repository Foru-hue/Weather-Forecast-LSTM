import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math, time
import re
import plotly.graph_objects as go

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 加载并预处理数据
def load_and_preprocess_data(filepath):
    try:
        # 读取CSV文件并解析日期列
        data = pd.read_csv(filepath, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"错误：未找到文件 {filepath}")
        return None, None

    # 处理数值列单位（去除℃和小时）
    data['Max Temp'] = data['Max Temp'].apply(lambda x: float(re.sub(r'℃', '', x)))
    data['Min Temp'] = data['Min Temp'].apply(lambda x: float(re.sub(r'℃', '', x)))
    data['Sunshine Duration'] = data['Sunshine Duration'].apply(lambda x: float(re.sub(r'小时', '', x)))

    # 提取最高温度和最低温度数据
    price = data[['Max Temp', 'Min Temp']].copy()

    # 处理缺失值（如果有）
    if price.isnull().any().any():
        price = price.interpolate(method='linear', limit_direction='both')
        print("提示：发现缺失值，已通过线性插值填充")

    # 可视化原始温度数据
    plt.figure(figsize=(15, 9))
    plt.plot(price[['Max Temp']], label='最高温度')
    plt.plot(price[['Min Temp']], label='最低温度')

    # 设置x轴日期标签（每30天显示一个日期）
    date_labels = data['Date'].dt.strftime('%Y-%m-%d').iloc[::30]
    plt.xticks(range(0, price.shape[0], 30), date_labels, rotation=45)

    # 设置图表标题和标签
    plt.title("邯郸最高和最低温度", fontsize=18, fontweight='bold')
    plt.xlabel('日期', fontsize=18)
    plt.ylabel('温度(℃)', fontsize=18)
    plt.legend()

    # 调整图表边距
    plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.15)
    plt.savefig("温度数据可视化.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 数据归一化处理（范围：-1到1）
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price = scaler.fit_transform(price)

    return price, scaler


# 分割数据为训练集和测试集
def split_data(stock, lookback):
    data = []
    # 构建滑动窗口数据集
    for index in range(len(stock) - lookback):
        data.append(stock[index: index + lookback])

    data = np.array(data)

    # 划分训练集和测试集（70%训练，30%测试）
    test_set_size = int(np.round(0.3 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    # 准备训练集和测试集数据
    x_train = data[:train_set_size, :-1, :]  # 输入序列（不包含最后一个时间步）
    y_train = data[:train_set_size, -1, :]  # 目标值（最后一个时间步）
    x_test = data[train_set_size:, :-1, :]
    y_test = data[train_set_size:, -1, :]

    return x_train, y_train, x_test, y_test, train_set_size, test_set_size


# 模型构建 定义LSTM模型类
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层（带dropout防止过拟合）
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)

        # 全连接层，将LSTM输出映射到目标维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # LSTM前向传播
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # 只取序列最后一个时间步的输出
        out = self.fc(out[:, -1, :])

        return out


# 训练模型
def train_model(model, x_train, y_train, x_test, y_test, num_epochs, lr):
    # 定义损失函数（均方误差）和优化器（Adam）
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 记录训练和测试损失
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    # 记录训练开始时间
    start_time = time.time()

    # 最佳验证损失初始化为无穷大
    best_val_loss = float('inf')

    # 训练循环
    for t in range(num_epochs):
        # 前向传播
        y_train_pred = model(x_train)
        train_loss = criterion(y_train_pred, y_train)

        # 记录训练损失
        train_hist[t] = train_loss.item()

        # 计算测试损失（不进行反向传播）
        with torch.no_grad():
            y_test_pred = model(x_test)
            test_loss = criterion(y_test_pred, y_test)
            test_hist[t] = test_loss.item()

        # 更新最佳验证损失
        if test_loss.item() < best_val_loss:
            best_val_loss = test_loss.item()

        # 每10个epoch或最后一个epoch打印训练信息
        if (t + 1) % 10 == 0 or (t + 1) == num_epochs:
            print(f"Epoch {t + 1}/{num_epochs} | 训练损失:{train_loss.item():.6f} | 验证损失:{test_loss.item():.6f}")

        # 反向传播和优化
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    # 计算训练耗时
    training_time = time.time() - start_time
    print(f"训练完成!耗时: {training_time:.2f} 秒 | 最佳验证损失:{best_val_loss:.6f}")

    return train_hist, test_hist


# 评估模型
def evaluate_model(model, x_train, y_train, x_test, y_test, scaler):
    # 获取模型预测
    y_train_pred = model(x_train)
    y_test_pred = model(x_test)

    # 反归一化处理，将预测值和实际值转换回原始温度范围
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train_actual = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test_actual = scaler.inverse_transform(y_test.detach().numpy())

    # 计算评估指标（RMSE和MAE）
    train_max_rmse = math.sqrt(mean_squared_error(y_train_actual[:, 0], y_train_pred[:, 0]))
    test_max_rmse = math.sqrt(mean_squared_error(y_test_actual[:, 0], y_test_pred[:, 0]))
    train_max_mae = mean_absolute_error(y_train_actual[:, 0], y_train_pred[:, 0])
    test_max_mae = mean_absolute_error(y_test_actual[:, 0], y_test_pred[:, 0])

    train_min_rmse = math.sqrt(mean_squared_error(y_train_actual[:, 1], y_train_pred[:, 1]))
    test_min_rmse = math.sqrt(mean_squared_error(y_test_actual[:, 1], y_test_pred[:, 1]))
    train_min_mae = mean_absolute_error(y_train_actual[:, 1], y_train_pred[:, 1])
    test_min_mae = mean_absolute_error(y_test_actual[:, 1], y_test_pred[:, 1])

    # 打印评估指标
    print("\n=== 最高温评估指标 ===")
    print(f"训练集: RMSE={train_max_rmse:.2f}℃ | MAE={train_max_mae:.2f}℃")
    print(f"测试集: RMSE={test_max_rmse:.2f}℃ | MAE={test_max_mae:.2f}℃")

    print("\n=== 最低温评估指标 ===")
    print(f"训练集: RMSE={train_min_rmse:.2f}℃ | MAE={train_min_mae:.2f}℃")
    print(f"测试集: RMSE={test_min_rmse:.2f}℃ | MAE={test_min_mae:.2f}℃")

    return y_train_pred, y_train_actual, y_test_pred, y_test_actual


# 绘制训练损失
def plot_training_loss(train_hist, test_hist):
    plt.figure(figsize=(12, 6))
    plt.plot(train_hist, 'b', label='训练损失')
    plt.plot(test_hist, 'r', label='测试损失')
    plt.xlabel("训练轮数", fontsize=14)
    plt.ylabel("损失值", fontsize=14)
    plt.title("训练和测试损失变化", fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("训练和测试损失.png", dpi=300, bbox_inches='tight')
    plt.show()


# 可视化预测结果
def visualize_predictions(data, y_train_pred, y_train_actual, y_test_pred, y_test_actual, lookback, scaler, price,
                          future_predictions):
    # 准备用于绘图的数据
    trainPredictPlot_max = np.empty_like(price[:, 0])
    trainPredictPlot_max[:] = np.nan
    trainPredictPlot_max[lookback:len(y_train_pred) + lookback] = y_train_pred[:, 0]

    trainPredictPlot_min = np.empty_like(price[:, 1])
    trainPredictPlot_min[:] = np.nan
    trainPredictPlot_min[lookback:len(y_train_pred) + lookback] = y_train_pred[:, 1]

    testPredictPlot_max = np.empty_like(price[:, 0])
    testPredictPlot_max[:] = np.nan
    testPredictPlot_max[len(y_train_pred) + lookback - 1:len(price) - 1] = y_test_pred[:, 0]

    testPredictPlot_min = np.empty_like(price[:, 1])
    testPredictPlot_min[:] = np.nan
    testPredictPlot_min[len(y_train_pred) + lookback - 1:len(price) - 1] = y_test_pred[:, 1]

    # 获取原始温度数据
    original_max = scaler.inverse_transform(price)[:, 0]
    original_min = scaler.inverse_transform(price)[:, 1]

    # 处理日期数据
    dates = pd.to_datetime(data['Date'])
    last_date = dates.iloc[-1]

    # 生成未来一周的日期
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_predictions))

    # 创建交互式图表
    fig = go.Figure()

    # 添加最高温度轨迹
    fig.add_trace(
        go.Scatter(x=dates, y=original_max, mode='lines', name='最高温 实际温度', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=trainPredictPlot_max, mode='lines', name='最高温 训练集预测',
                             line=dict(color='orange', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=dates, y=testPredictPlot_max, mode='lines', name='最高温 测试集预测',
                             line=dict(color='green', width=2)))
    # 添加最低温度轨迹
    fig.add_trace(
        go.Scatter(x=dates, y=original_min, mode='lines', name='最低温 实际温度', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=trainPredictPlot_min, mode='lines', name='最低温 训练集预测',
                             line=dict(color='purple', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=dates, y=testPredictPlot_min, mode='lines', name='最低温 测试集预测',
                             line=dict(color='brown', width=2)))
    # 添加未来预测轨迹
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions[:, 0], mode='lines', name='最高温 未来一周预测',
                             line=dict(color='cyan', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions[:, 1], mode='lines', name='最低温 未来一周预测',
                             line=dict(color='magenta', width=2)))
    # 设置图表布局
    fig.update_layout(
        title={"text": "邯郸最高和最低温度预测", "x": 0.5, "xanchor": "center"},
        xaxis_title="时间",
        yaxis_title="温度(℃)",
        legend_title="图例",
        font=dict(size=14),
        hovermode="x unified",  # 鼠标悬停时显示所有轨迹的数据
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d',  # 日期格式
            dtick='M1',  # 每月显示一个刻度
            tickangle=45,
            # 设置x轴范围（包含历史数据和未来预测）
            range=[dates.min() - pd.DateOffset(months=1), future_dates.max() + pd.DateOffset(months=1)]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),  # 图例位置
        margin=dict(l=80, r=80, b=80, t=80, pad=4)  # 边距
    )
    # 显示图表
    fig.show()


# 预测未来一周天气
def predict_future_weather(model, price, lookback, scaler, num_predictions=10):
    future_steps = 7  # 预测未来7天
    all_predictions = []

    # 多次预测并取平均，提高预测稳定性
    for _ in range(num_predictions):
        future_predictions = []
        last_sequence = price[-lookback:].reshape(1, lookback, 2)
        last_sequence = torch.from_numpy(last_sequence).type(torch.Tensor)

        # 逐步预测未来温度
        for _ in range(future_steps):
            next_step = model(last_sequence)
            future_predictions.append(next_step.detach().numpy()[0])

            # 更新序列，移除最早的时间步，添加最新预测
            new_sequence = torch.cat((last_sequence[:, 1:, :], next_step.unsqueeze(1)), dim=1)
            last_sequence = new_sequence

        all_predictions.append(future_predictions)

    # 计算多次预测的平均值
    average_predictions = np.mean(all_predictions, axis=0)

    # 反归一化，转换回原始温度范围
    average_predictions = scaler.inverse_transform(average_predictions)

    return average_predictions


def main():
    filepath = './handan.csv'
    price, scaler = load_and_preprocess_data(filepath)
    if price is None:
        return
    data = pd.read_csv(filepath, parse_dates=['Date'])
    print(f"读入了 {len(data)} 条数据")

    # 数据分割
    lookback = 14  # 时间窗口大小
    x_train, y_train, x_test, y_test, train_set_size, test_set_size = split_data(price, lookback)

    # 输出训练集和测试集大小
    print(f"训练集大小:{train_set_size} | 测试集大小:{test_set_size}")

    # 转换为PyTorch张量
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    # 模型配置
    input_dim = 2  # 输入特征维度（最高温度和最低温度）
    hidden_dim = 128  # 隐藏层维度
    num_layers = 3  # LSTM层数
    output_dim = 2  # 输出维度（预测最高温度和最低温度）
    num_epochs = 150  # 训练轮数
    lr = 0.01  # 学习率

    # 输出模型配置
    print(f"模型配置:隐藏层维度={hidden_dim}, LSTM层数={num_layers},训练轮数={num_epochs}")

    # 初始化模型
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    print(model)

    # 训练模型
    train_hist, test_hist = train_model(model, x_train, y_train, x_test, y_test, num_epochs, lr)

    # 可视化训练损失
    plot_training_loss(train_hist, test_hist)

    # 评估模型
    y_train_pred, y_train_actual, y_test_pred, y_test_actual = evaluate_model(model, x_train, y_train, x_test, y_test,
                                                                              scaler)

    # 预测未来一周的天气
    future_predictions = predict_future_weather(model, price, lookback, scaler, num_predictions=10)
    print("\n未来一周最高温度和最低温度预测：")
    print(future_predictions)

    # 可视化预测结果
    visualize_predictions(data, y_train_pred, y_train_actual, y_test_pred, y_test_actual, lookback, scaler, price,
                          future_predictions)


if __name__ == "__main__":
    main()
