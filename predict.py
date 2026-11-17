"""读取保存的北京模型，演示历史下一小时气温预测。"""
import json
import argparse
from pathlib import Path
from weather_model import predict

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='北京官园站下一小时气温预测')
    parser.add_argument('--timestamp',default='2016-05-12T19:00')
    args=parser.parse_args()
    model=json.loads((ROOT/'outputs/model.json').read_text(encoding='utf-8'))
    text=(ROOT/'outputs/demo-data.js').read_text(encoding='utf-8')
    data=json.loads(text[len('window.BEIJING_DATA = '):-1])
    row=next((r for r in data['inputs'] if r['timestamp']==args.timestamp),None)
    if row is None or row['temp_prev'] is None: parser.error('该时刻缺少有效历史输入。')
    print(f"{args.timestamp}，预计气温 {predict(model,row):.2f}℃；沿用当前气温对照为 {row['temp_prev']:.2f}℃。")
