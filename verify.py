"""验证预测信息时间边界、目标缺测、指标计算和独立运行公式。"""
import json
import numpy as np
import pandas as pd
from main import ROOT,OUT,read_data,make_features
from weather_model import predict

df,_=read_data();x=make_features(df)
model=json.loads((OUT/'model.json').read_text(encoding='utf-8'))
metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
result=pd.read_csv(OUT/'test_predictions.csv',parse_dates=['timestamp'])
original=df.set_index('timestamp')
assert model['training_end']<result.timestamp.min().isoformat(timespec='minutes')
errors=[];baseline_errors=[]
for row in result.itertuples():
    observed=original.loc[row.timestamp,'TEMP']
    assert (pd.isna(observed) and pd.isna(row.actual)) or observed==row.actual
    idx=int(df.index[df.timestamp.eq(row.timestamp)][0])
    values={k:None if pd.isna(v) else float(v) for k,v in x.loc[idx].items()}
    if pd.notna(row.prediction): assert abs(predict(model,values)-row.prediction)<1e-8
    if row.evaluated:
        assert pd.notna(observed) and pd.notna(row.prediction) and pd.notna(row.persistence)
        errors.append(abs(observed-row.prediction));baseline_errors.append(abs(observed-row.persistence))
assert len(errors)==metrics['evaluated_hours']
assert abs(np.mean(errors)-metrics['scores']['forecast']['MAE'])<1e-9
assert abs(np.mean(baseline_errors)-metrics['scores']['persistence']['MAE'])<1e-9
# 修改目标时刻及未来温度，不应影响该时刻及之前任何输入。
index=30000
changed=df.copy();changed.loc[index:,'TEMP']+=1000
after=make_features(changed)
assert x.iloc[:index+1].equals(after.iloc[:index+1])
assert np.isclose(after.loc[index+1,'temp_prev']-x.loc[index+1,'temp_prev'],1000,atol=1e-9,rtol=0)
validation=pd.read_csv(OUT/'validation_results.csv')
assert len(validation)==6
assert all(validation.fit_end<validation.validation_start)
assert all(validation.validation_end<=model['training_end'])
assert metrics['selected']==min(metrics['validation_mean_mae'],key=metrics['validation_mean_mae'].get)
print(f'PASS: {len(result)} test hours checked; past-only inputs, missing targets, metrics, validation and exported formula agree.')
