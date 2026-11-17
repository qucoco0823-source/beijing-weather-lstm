"""用已保存的LSTM权重演示历史下一小时预测。"""
import argparse
import json
import numpy as np
from main import ROOT,read_data
from lstm_model import make_sequences,numpy_predict


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='北京官园站LSTM下一小时气温预测')
    parser.add_argument('--timestamp',default='2016-05-12T19:00')
    args=parser.parse_args()
    path=ROOT/'outputs/lstm/model.json'
    if not path.exists():parser.error('请先运行LSTM训练。')
    saved=json.loads(path.read_text(encoding='utf-8'))
    frame,_=read_data()
    targets=np.flatnonzero(frame.timestamp.eq(args.timestamp))
    if not len(targets):parser.error('没有这个历史时刻。')
    x,indices,last,_=make_sequences(frame,saved['scaler'])
    matches=np.flatnonzero(indices==targets[0])
    if not len(matches):parser.error('此前24小时历史不完整，不能预测。')
    j=matches[0]
    value=last[j]+numpy_predict(saved,x[j])*saved['scaler']['delta_std']
    print(f'{args.timestamp}：LSTM预计气温 {value:.2f}℃，使用此前24小时历史，不读取目标气温。')
