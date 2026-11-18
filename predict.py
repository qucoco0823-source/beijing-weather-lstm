"""读取已保存参数做一个测试年历史预测，不重新训练。"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch
from train_models import ROOT,make_features,DailyLSTM

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--date',default='2025-01-01');parser.add_argument('--model',choices=['ridge','forest','lstm']);args=parser.parse_args()
    df=pd.read_csv(ROOT/'data/cleaned_weather.csv',parse_dates=['date'])
    rows=df.index[df.date.eq(args.date)]
    if not len(rows):parser.error('没有这个历史日期。')
    i=int(rows[0]);x=make_features(df).iloc[[i]]
    if x.isna().any().any():parser.error('历史输入不完整。')
    bundle=json.loads((ROOT/'outputs/models.json').read_text(encoding='utf-8'));name=args.model or bundle['selected']
    if name=='lstm':
        past=df.iloc[i-7:i];s=bundle['lstm']['scaler'];high=past.high.to_numpy();low=past.low.to_numpy();cycle=2*np.pi*(past.date.dt.dayofyear.to_numpy()-1)/365.25
        sequence=np.column_stack(((high-s['high_mean'])/s['high_std'],(low-s['low_mean'])/s['low_std'],np.diff(high,prepend=high[0])/s['delta_std'],np.sin(cycle),np.cos(cycle))).astype(np.float32)
        checkpoint=torch.load(ROOT/'outputs/lstm_checkpoint.pt',map_location='cpu',weights_only=True);model=DailyLSTM();model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.no_grad():value=float(high[-1]+model(torch.from_numpy(sequence[None])).item()*s['delta_std'])
    else:value=float(joblib.load(ROOT/'outputs'/f'{name}.joblib').predict(x.to_numpy())[0])
    print(f'{args.date} {name}预计网页最高气温：{value:.2f}℃；只使用此前7天历史。')
