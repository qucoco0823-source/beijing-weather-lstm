# 北京天气自主采集、分类分析与气温预测

本版本的主流程全部使用自主采集的网页记录：请求月份HTML、解析每日表格、清洗去重、信息分类、探索分析，再训练下一天最高气温模型。

## 数据来源

- 来源：天气后报北京历史天气公开月份网页，https://www.tianqihoubao.com/lishi/beijing.html 。
- 采集范围：2021年1月至2025年12月，共60个月份网页。
- 网站说明数据源于城市当天的天气预报信息。本项目处理的是网站历史归档，不将它称作独立气象实测。
- 原始HTML及SHA-256保存在data/raw_html和data/crawl_manifest.json，逐请求日志在data/crawl_log.jsonl。
- 采集器使用urllib和HTMLParser，不是自动下载现成CSV。不执行网页脚本，不使用登录账户或密钥，不绕过拒绝访问和限流。
- robots规则页实际返回406，程序记录该响应。仅低频串行读取公开月份页。若收到401、403或429会停止。

## 运行

在安装依赖的Python环境下，从本目录执行：

```sh
python -m pip install -r requirements.txt
python crawl_weather.py
python clean_analyze.py
python train_models.py
```

已有缓存时复用HTML。只有需要重新发出月份请求时才加 `--refresh`。如本机网络需要代理，可传 `--proxy http://127.0.0.1:7890`，程序不把代理写进数据。

打开index.html查看离线演示。重新训练只读本项目data/crawled_raw.csv，不读取以前下载的数据集。

## 文件与作用

- crawl_weather.py：按年月采集、缓存原始HTML、解析weather-table、校验日历日期、保存原始行与来源。
- clean_analyze.py：原文保留、重复核查、温度拆分、范围检查、天气分类、月份和季节统计。
- train_models.py：历史7天特征、按年份切分、岭回归、浅随机森林与单层LSTM、共同测试集评价、模型导出。
- outputs/metrics.json：验证选择依据、实际测试指标和数据边界。
- outputs/models.json及模型文件：真实训练结果，浏览器使用相同参数计算。
- verify_ui.cjs：逐个比对三种模型在测试年所有有效日期的网页推理，核查原始网页摘要和分类总数，检查未来目标不影响输入。

## 统计与预测口径

分类是确定的文字规则，一天归入一个类别；雨、雪标记另行统计。网页没有降水毫米数，不生成降水量结论。
2021—2023年拟合，2024年验证选择模型和LSTM轮数，在2021—2024年重训，2025年整年测试。标准化统计只使用拟合段。
目标是下一天的网页最高气温。模型只读取目标之前的记录，真实目标不补。评价逐日使用此前已归档记录，不是一次连续预测未来全年。

这是课程中的单网站历史归档分析，不是实时天气预报服务。

在线演示：https://qucoco0823-source.github.io/beijing-weather-lstm/
