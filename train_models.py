"""只用自主爬到的日天气训练，按完整年份留出检验下一天最高气温。"""
import copy
import json
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from clean_analyze import ROOT,clean,analyze

WINDOW=7
FEATURE_NAMES=['前一天最高温','前一天最低温','前一天昼夜温差','前一天最高温变化','过去7天最高温均值','过去7天最高温标准差','过去7天最低最高温','过去7天最高最高温','目标日年周期正弦','目标日年周期余弦','前一天有雨','前一天有雪','前一天风力等级']


class DailyLSTM(nn.Module):
    def __init__(self):
        super().__init__();self.lstm=nn.LSTM(5,16,batch_first=True);self.output=nn.Linear(16,1)
    def forward(self,x):
        sequence,_=self.lstm(x);return self.output(sequence[:,-1]).squeeze(-1)


def metrics(y,p):
    return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),'days':len(y)}


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)


def make_features(df):
    h=df.high;l=df.low;cycle=2*np.pi*(df.date.dt.dayofyear-1)/365.25
    table=pd.DataFrame({'previous_high':h.shift(1),'previous_low':l.shift(1),'previous_range':(h-l).shift(1),'previous_change':h.diff().shift(1),'mean_7':h.shift(1).rolling(WINDOW).mean(),'std_7':h.shift(1).rolling(WINDOW).std(ddof=0),'min_7':h.shift(1).rolling(WINDOW).min(),'max_7':h.shift(1).rolling(WINDOW).max(),'year_sin':np.sin(cycle),'year_cos':np.cos(cycle),'previous_rain':df.rain_flag.shift(1),'previous_snow':df.snow_flag.shift(1),'previous_wind':df.wind_level.shift(1)})
    return table


def train_lstm(x,y,fit_mask,val_mask,epochs=60,patience=8):
    seed();model=DailyLSTM();optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-4)
    train_x=torch.from_numpy(x[fit_mask]);train_y=torch.from_numpy(y[fit_mask]);valid_x=torch.from_numpy(x[val_mask]);valid_y=torch.from_numpy(y[val_mask])
    history=[];best=float('inf');best_epoch=1;best_state=None;stale=0
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for batch in torch.randperm(len(train_x)).split(64):
            optimizer.zero_grad();loss=nn.functional.mse_loss(model(train_x[batch]),train_y[batch]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():val=float((model(valid_x)-valid_y).abs().mean())
        history.append({'epoch':epoch,'train_mse_standardized_delta':float(np.mean(losses)),'validation_mae_standardized_delta':val})
        if val<best-1e-4:best=val;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best_state);return model,best_epoch,history


def refit_lstm(x,y,mask,epochs):
    seed();model=DailyLSTM();optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-4)
    xt=torch.from_numpy(x[mask]);yt=torch.from_numpy(y[mask])
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(xt)).split(64):
            optimizer.zero_grad();loss=nn.functional.mse_loss(model(xt[batch]),yt[batch]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
    model.eval();return model


def train():
    df,audit=clean();analysis=analyze(df)
    # 日历对齐防止缺失日期被当作相邻天。这里不填补任何目标。
    frame=df.set_index('date').reindex(pd.date_range(df.date.min(),df.date.max())).rename_axis('date').reset_index()
    features=make_features(frame)
    early=frame.date<'2024-01-01';valid=(frame.date>='2024-01-01')&(frame.date<'2025-01-01');test=frame.date>='2025-01-01'
    stats=frame.loc[early,['high','low']].agg(['mean','std']);delta_std=float(frame.loc[early,'high'].diff().std(ddof=0))
    scale={'high_mean':float(stats.loc['mean','high']),'high_std':float(frame.loc[early,'high'].std(ddof=0)),'low_mean':float(stats.loc['mean','low']),'low_std':float(frame.loc[early,'low'].std(ddof=0)),'delta_std':delta_std,'fit_end':'2023-12-31'}
    sequence=[];positions=[]
    for i in range(WINDOW,len(frame)):
        past=frame.iloc[i-WINDOW:i]
        if past[['high','low']].isna().any().any():continue
        h=past.high.to_numpy();l=past.low.to_numpy();cycle=2*np.pi*(past.date.dt.dayofyear.to_numpy()-1)/365.25
        sequence.append(np.column_stack(((h-scale['high_mean'])/scale['high_std'],(l-scale['low_mean'])/scale['low_std'],np.diff(h,prepend=h[0])/delta_std,np.sin(cycle),np.cos(cycle))))
        positions.append(i)
    positions=np.asarray(positions);x=np.asarray(sequence,dtype=np.float32);tab=features.iloc[positions].to_numpy(dtype=float);target=frame.high.to_numpy()[positions];previous=frame.high.shift(1).to_numpy()[positions]
    y=((target-previous)/delta_std).astype(np.float32)
    available=np.isfinite(tab).all(axis=1)&np.isfinite(y)
    fit_mask=early.to_numpy()[positions]&available;val_mask=valid.to_numpy()[positions]&available;test_mask=test.to_numpy()[positions]&available;full_mask=(~test.to_numpy()[positions])&available
    ridge=make_pipeline(StandardScaler(),Ridge(alpha=10))
    forest=RandomForestRegressor(n_estimators=120,max_depth=6,min_samples_leaf=6,random_state=42,n_jobs=2)
    models={'ridge':ridge,'forest':forest};validation_scores={}
    for name,model in models.items():
        model.fit(tab[fit_mask],target[fit_mask]);validation_scores[name]=metrics(target[val_mask],model.predict(tab[val_mask]))
    lstm,epochs,history=train_lstm(x,y,fit_mask,val_mask)
    with torch.no_grad():val_prediction=previous[val_mask]+lstm(torch.from_numpy(x[val_mask])).numpy()*delta_std
    validation_scores['lstm']=metrics(target[val_mask],val_prediction)
    validation_scores['persistence']=metrics(target[val_mask],previous[val_mask])
    selected=min(['ridge','forest','lstm'],key=lambda k:validation_scores[k]['MAE'])
    print('2024年验证MAE：', {k:round(v['MAE'],3) for k,v in validation_scores.items()},'选定：',selected,flush=True)
    for model in models.values():model.fit(tab[full_mask],target[full_mask])
    lstm=refit_lstm(x,y,full_mask,epochs)
    outputs={k:model.predict(tab[test_mask]) for k,model in models.items()};outputs['persistence']=previous[test_mask]
    with torch.no_grad():outputs['lstm']=previous[test_mask]+lstm(torch.from_numpy(x[test_mask])).numpy()*delta_std
    scores={k:metrics(target[test_mask],prediction) for k,prediction in outputs.items()}
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    result=pd.DataFrame({'date':frame.date.iloc[positions[test_mask]].to_numpy(),'actual':target[test_mask],**outputs});result.to_csv(out/'test_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    weights={key:value.detach().numpy().tolist() for key,value in lstm.state_dict().items()}
    exported_lstm={'window':WINDOW,'input_size':5,'hidden_size':16,'scaler':scale,'weights':weights,'selected_epochs':epochs}
    torch.save({'state_dict':lstm.state_dict(),'metadata':exported_lstm},out/'lstm_checkpoint.pt')
    for name,model in models.items():joblib.dump(model,out/f'{name}.joblib')
    tree_data=[]
    for estimator in forest.estimators_:
        tree=estimator.tree_;tree_data.append({'left':tree.children_left.tolist(),'right':tree.children_right.tolist(),'feature':tree.feature.tolist(),'threshold':tree.threshold.tolist(),'value':tree.value[:,0,0].tolist()})
    exported_ridge={'mean':ridge[0].mean_.tolist(),'scale':ridge[0].scale_.tolist(),'coefficient':ridge[1].coef_.tolist(),'intercept':float(ridge[1].intercept_)}
    model_bundle={'selected':selected,'ridge':exported_ridge,'forest':tree_data,'lstm':exported_lstm,'feature_names':FEATURE_NAMES}
    report={'selected_model':selected,'selection_rule':'只按2024年验证MAE选择，模型与设置在查看2025年测试成绩前固定','fit_period':'2021-01-01至2023-12-31','validation_period':'2024-01-01至2024-12-31','refit_period':'2021-01-01至2024-12-31','test_period':'2025-01-01至2025-12-31','test_calendar_days':int(test.sum()),'evaluated_days':int(test_mask.sum()),'excluded_test_days':int(test.sum()-test_mask.sum()),'window_days':WINDOW,'lstm_epochs':epochs,'validation_scores':validation_scores,'scores':scores,'test_mode':'逐日使用此前已归档记录，预测下一天网页最高气温；不是从起点连续预测一年','target':'网站历史天气归档中的下一天最高气温；非独立气象实测验证','feature_columns':list(features.columns),'seed':42}
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'models.json').write_text(json.dumps(model_bundle,ensure_ascii=False),encoding='utf-8')
    manifest=json.loads((ROOT/'data/crawl_manifest.json').read_text(encoding='utf-8'))
    payload={'source':manifest,'audit':audit,'analysis':analysis,'metrics':report,'models':model_bundle,'history':history,'records':json.loads(df.assign(date=df.date.dt.strftime('%Y-%m-%d')).to_json(orient='records')),'predictions':json.loads(result.assign(date=result.date.dt.strftime('%Y-%m-%d')).to_json(orient='records',double_precision=10))}
    (out/'demo-data.js').write_text('window.CRAWLED_WEATHER = '+json.dumps(payload,ensure_ascii=False,allow_nan=False)+';',encoding='utf-8')
    print('2025年共同测试集：',json.dumps(scores,ensure_ascii=False),flush=True)
    return payload


if __name__=='__main__':train()
