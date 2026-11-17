"""验证真实checkpoint、独立公式、时间边界和公平比较。"""
import json
import numpy as np
import pandas as pd
import torch
from main import ROOT,read_data,scores
from lstm_model import TemperatureLSTM,make_sequences,numpy_predict,fit_scaler

out=ROOT/'outputs/lstm'
saved=json.loads((out/'model.json').read_text(encoding='utf-8'))
metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
frame,_=read_data();split=int(len(frame)*.8);fit_end=int(split*.8)
assert saved['training_end']<metrics['test_start']
assert saved['scaler_fit_end']<metrics['validation_start']
assert saved['scaler']==fit_scaler(frame,fit_end)
x,idx,last,delta=make_sequences(frame,saved['scaler'])
checkpoint=torch.load(out/'lstm_checkpoint.pt',map_location='cpu',weights_only=True)
model=TemperatureLSTM();model.load_state_dict(checkpoint['state_dict']);model.eval();torch.set_num_threads(4)
positions=np.flatnonzero(idx>=split)
with torch.inference_mode():
    predicted_delta=np.concatenate([model(batch).numpy() for batch in torch.split(torch.from_numpy(x[positions]),512)])
actual_prediction=last[positions]+predicted_delta*saved['scaler']['delta_std']
result=pd.read_csv(out/'test_predictions.csv')
assert np.allclose(result.loc[idx[positions]-split,'lstm_prediction'],actual_prediction,atol=1e-7,rtol=0)
for j in np.linspace(0,len(positions)-1,40,dtype=int):
    assert abs(numpy_predict(saved,x[positions[j]])-float(predicted_delta[j]))*saved['scaler']['delta_std']<1e-4
mask=result.fair_evaluation
for key,col in [('lstm','lstm_prediction'),('linear','linear_prediction'),('persistence','persistence')]:
    measured=scores(result.actual.where(mask),result[col])
    assert measured['hours']==metrics['evaluated_hours']
    assert abs(measured['MAE']-metrics['scores'][key]['MAE'])<1e-9
# 验证真实目标修改不影响同一时刻的输入窗口，下一时刻才可读到。
small=frame.iloc[:100].copy();changed=small.copy();changed.loc[50,'TEMP']+=100
before,indices,_,_=make_sequences(small,saved['scaler'])
after,changed_indices,_,_=make_sequences(changed,saved['scaler'])
assert np.array_equal(indices,changed_indices)
assert np.array_equal(before[indices==50],after[indices==50])
assert not np.array_equal(before[indices==51],after[indices==51])
print('PASS: checkpoint reproduces saved predictions; scaler/early stopping dates, past-only windows, independent LSTM and fair metrics verified.')
