"""便携源码包入口：已有爬取CSV则直接清洗分析训练，否则先采集。"""
from crawl_weather import ROOT,crawl
from train_models import train
if __name__=='__main__':
    if not (ROOT/'data/crawled_raw.csv').exists():crawl()
    train()
