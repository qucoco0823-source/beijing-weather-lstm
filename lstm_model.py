"""单层LSTM读取过去24小时，预测下一小时温度变化。"""
import numpy as np
import torch
from torch import nn

WINDOW=24
HIDDEN=32


class TemperatureLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(input_size=4,hidden_size=HIDDEN,num_layers=1,batch_first=True)
        self.output=nn.Linear(HIDDEN,1)

    def forward(self,x):
        sequence,_=self.lstm(x)
        return self.output(sequence[:,-1,:]).squeeze(-1)


def fit_scaler(frame,end):
    # end为拟合子集的排他边界，不读取验证或测试统计量。
    temperature=frame.TEMP.iloc[:end].ffill(limit=3)
    return {'temperature_mean':float(temperature.mean()),'temperature_std':float(temperature.std(ddof=0)),
            'delta_std':float(temperature.diff().std(ddof=0))}


def make_sequences(frame,scaler):
    temperature=frame.TEMP.ffill(limit=3).to_numpy(dtype=np.float64)
    hours=frame.hour.to_numpy()
    windows=[];targets=[];last=[];delta=[]
    # i为目标时刻。窗口严格是[i-24,i)，不包括目标温度。
    for i in range(WINDOW,len(frame)):
        past=temperature[i-WINDOW:i]
        if not np.isfinite(past).all(): continue
        change=np.diff(past,prepend=past[0])/scaler['delta_std']
        h=hours[i-WINDOW:i]
        features=np.column_stack(((past-scaler['temperature_mean'])/scaler['temperature_std'],
                                  change,np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24)))
        windows.append(features);targets.append(i);last.append(past[-1])
        true=frame.TEMP.iloc[i]
        delta.append((true-past[-1])/scaler['delta_std'] if np.isfinite(true) else np.nan)
    return np.asarray(windows,dtype=np.float32),np.asarray(targets),np.asarray(last),np.asarray(delta,dtype=np.float32)


def export_model(model,scaler):
    state=model.state_dict()
    return {'model_type':'lstm','window':WINDOW,'hidden_size':HIDDEN,'input_size':4,'layers':1,
        'input_names':['标准化气温','窗口内标准化温差','已知小时正弦','已知小时余弦'],
        'output_mode':'前1小时气温 + LSTM预测温差','scaler':scaler,
        'weights':{k:v.detach().cpu().tolist() for k,v in state.items()}}


def numpy_predict(saved,sequence):
    """独立重现PyTorch的i/f/g/o四门公式，用于模型导出检验。"""
    w=saved['weights'];size=saved['hidden_size']
    wi=np.asarray(w['lstm.weight_ih_l0']);wh=np.asarray(w['lstm.weight_hh_l0'])
    bias=np.asarray(w['lstm.bias_ih_l0'])+np.asarray(w['lstm.bias_hh_l0'])
    h=np.zeros(size);c=np.zeros(size)
    for x in sequence:
        gates=wi@x+wh@h+bias
        i,f,g,o=np.split(gates,4)
        sigmoid=lambda a:1/(1+np.exp(-np.clip(a,-50,50)))
        c=sigmoid(f)*c+sigmoid(i)*np.tanh(g)
        h=sigmoid(o)*np.tanh(c)
    return float(np.asarray(w['output.weight'])[0]@h+w['output.bias'][0])
