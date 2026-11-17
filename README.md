# 北京天气分析与LSTM下一小时气温预测

课程项目初始版本：实际天气记录 → 清洗检查 → 历史分析 → LSTM训练 → 公平评估 → 交互演示。

## 在线演示

本仓库根目录是静态网站。GitHub Pages 启用后可以直接分享网页链接。
浏览器读取导出的真实网络权重并完成预测；GitHub Pages 不执行 Python 训练。
打开 index.html 也可以离线查看，不需要账号和密钥。

## 运行与复现

使用支持 PyTorch 的 Python 环境，在仓库目录执行：

```sh
python -m pip install -r requirements.txt
python main.py
python train_lstm.py
python verify_lstm.py
python predict_lstm.py
```

main.py 生成线性对照及图表；train_lstm.py 训练 LSTM，并更新页面所用数据和模型权重。
网页数值检查可运行 `node verify_lstm_ui.cjs`。

## 当前结果

同一批 6947 个有效测试时刻：

| 方法 | MAE（℃） | RMSE（℃） |
| --- | ---: | ---: |
| LSTM | 0.553 | 0.825 |
| 线性回归 | 0.609 | 0.887 |
| 沿用前1小时气温 | 0.841 | 1.151 |

单层 LSTM、32 个隐藏单元，读取此前24小时，预测下一小时温差。
时间顺序划分训练和测试，训练内部验证选择轮数；测试期不参与选轮数。
这是使用此前实际观测的逐小时历史回放，不是实时天气服务，也不代表一次性预测数月的效果。

## 文件

- index.html、style.css、app.js、lstm-runtime.js、lstm-ui.js：演示界面和浏览器推理。
- main.py、weather_model.py：数据清洗、分析和线性对照。
- train_lstm.py、lstm_model.py：真实 LSTM 训练与序列构建。
- outputs/lstm：权重、训练记录、指标、预测明细和图表。
- data/original_stations：官园站原始数据，保留缺测记录。

## 数据来源与许可

Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository.
DOI: [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G)。
[官方数据页面](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)。
数据依据 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 使用。
本项目选择官园站 2013-03-01 至 2017-02-28 的天气字段，派生出清洗记录、图表和预测结果；不使用污染物预测天气。
原始文件来自公开数据集，不包含本地账号、密码或密钥。
