"""LSTM训练与预测分析：时间切分、验证早停、重训及同条件对照。"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import read_data,scores
from lstm_model import TemperatureLSTM,fit_scaler,make_sequences,export_model,numpy_predict,WINDOW,HIDDEN

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/lstm'


def seed_all():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(4)


def run_epoch(model,loader,optimizer=None):
    model.train(optimizer is not None)
    total=0.;count=0
    for x,y in loader:
        with torch.set_grad_enabled(optimizer is not None):
            pred=model(x);loss=nn.functional.mse_loss(pred,y)
            if optimizer is not None:
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        total+=float(loss.detach())*len(x);count+=len(x)
    return total/count


def predict_batches(model,x):
    model.eval();result=[]
    with torch.inference_mode():
        for batch in torch.split(torch.from_numpy(x),512): result.append(model(batch).numpy())
    return np.concatenate(result)


def train(max_epochs=30,patience=6):
    OUT.mkdir(parents=True,exist_ok=True);seed_all()
    frame,audit=read_data();split=int(len(frame)*.8);fit_end=int(split*.8)
    scaler=fit_scaler(frame,fit_end)
    x,indices,last,delta=make_sequences(frame,scaler)
    observed=np.isfinite(delta)
    fit_mask=(indices<fit_end)&observed
    val_mask=(indices>=fit_end)&(indices<split)&observed
    full_train=(indices<split)&observed
    test_mask=indices>=split
    def loader(mask,shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(x[mask]),torch.from_numpy(delta[mask])),batch_size=128,shuffle=shuffle,num_workers=0)
    training=loader(fit_mask,True);validation=loader(val_mask,False)
    model=TemperatureLSTM();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    log=[];best=float('inf');best_epoch=0;stale=0
    start=time.perf_counter()
    print(f'LSTM：过去{WINDOW}小时 → 下一小时；单层{HIDDEN}隐藏单元；CPU训练。',flush=True)
    print(f'拟合{int(fit_mask.sum())}窗，验证{int(val_mask.sum())}窗，测试期在训练结束前不计算指标。',flush=True)
    for epoch in range(1,max_epochs+1):
        train_mse=run_epoch(model,training,optimizer)
        val_mse=run_epoch(model,validation)
        val_delta=predict_batches(model,x[val_mask])
        val_mae=float(np.mean(abs(val_delta-delta[val_mask]))*scaler['delta_std'])
        log.append({'epoch':epoch,'train_mse_standardized_delta':train_mse,'validation_mse_standardized_delta':val_mse,
                    'validation_mae_c':val_mae})
        pd.DataFrame(log).to_csv(OUT/'training_history.csv',index=False,encoding='utf-8-sig')
        print(f'验证选轮数 {epoch:02d}/{max_epochs}：训练MSE={train_mse:.4f}，验证MAE={val_mae:.4f}℃',flush=True)
        if val_mae<best-1e-4:
            best=val_mae;best_epoch=epoch;stale=0
        else:
            stale+=1
            if stale>=patience: break
    # 选好轮数后重新初始化，在全部前80%数据上训练；不用测试期选checkpoint。
    print(f'验证选择{best_epoch}轮，开始用完整训练期重训。',flush=True)
    seed_all();model=TemperatureLSTM();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    full_loader=loader(full_train,True)
    final_log=[]
    for epoch in range(1,best_epoch+1):
        mse=run_epoch(model,full_loader,optimizer);final_log.append({'epoch':epoch,'mse':mse})
        print(f'完整训练 {epoch:02d}/{best_epoch}：MSE={mse:.4f}',flush=True)
    pd.DataFrame(final_log).to_csv(OUT/'final_training_history.csv',index=False,encoding='utf-8-sig')
    saved=export_model(model,scaler)
    saved.update(seed=42,batch_size=128,learning_rate=.001,selected_epochs=best_epoch,
        validation_best_mae=best,scaler_fit_end=frame.timestamp.iloc[fit_end-1].isoformat(timespec='minutes'),
        training_end=frame.timestamp.iloc[split-1].isoformat(timespec='minutes'))
    torch.save({'state_dict':model.state_dict(),'metadata':saved},OUT/'lstm_checkpoint.pt')
    (OUT/'model.json').write_text(json.dumps(saved,ensure_ascii=False),encoding='utf-8')
    test_indices=indices[test_mask];test_delta=predict_batches(model,x[test_mask])
    test_prediction=last[test_mask]+test_delta*scaler['delta_std']
    result=pd.read_csv(ROOT/'outputs/test_predictions.csv',parse_dates=['timestamp']).rename(columns={'prediction':'linear_prediction','absolute_error':'linear_absolute_error'})
    result['lstm_prediction']=np.nan
    positions=test_indices-split
    result.loc[positions,'lstm_prediction']=test_prediction
    fair=result.actual.notna()&result.lstm_prediction.notna()&result.linear_prediction.notna()&result.persistence.notna()
    y=result.actual.where(fair)
    comparison={name:scores(y,result[col]) for name,col in [('lstm','lstm_prediction'),('linear','linear_prediction'),('persistence','persistence')]}
    result['lstm_absolute_error']=abs(result.actual-result.lstm_prediction)
    result['fair_evaluation']=fair
    result.to_csv(OUT/'test_predictions.csv',index=False,encoding='utf-8-sig')
    metrics={'scores':comparison,'evaluated_hours':int(fair.sum()),'test_hours':len(result),'excluded_hours':int((~fair).sum()),
        'window':WINDOW,'hidden_size':HIDDEN,'selected_epochs':best_epoch,'max_epochs':max_epochs,'patience':patience,
        'validation_best_mae':best,'fit_windows':int(full_train.sum()),'elapsed_seconds':time.perf_counter()-start,
        'training_end':saved['training_end'],'validation_start':frame.timestamp.iloc[fit_end].isoformat(timespec='minutes'),
        'test_start':result.timestamp.min().isoformat(timespec='minutes'),'test_end':result.timestamp.max().isoformat(timespec='minutes'),
        'forecast_mode':'rolling_one_hour_ahead','torch_version':torch.__version__,'device':'cpu'}
    (OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    # 分布式地抽查导出权重，覆盖全部测试期，不只检查第一个样本。
    for j in np.linspace(0,len(test_indices)-1,40,dtype=int):
        k=np.flatnonzero(test_mask)[j]
        assert abs(numpy_predict(saved,x[k])-float(test_delta[j]))*scaler['delta_std']<1e-4
    draw_plots(result,pd.DataFrame(log))
    payload={'model':saved,'metrics':metrics,'predictions':json.loads(result.assign(timestamp=result.timestamp.dt.strftime('%Y-%m-%dT%H:%M')).to_json(orient='records',double_precision=12)),
        'history':log}
    (OUT/'demo-data.js').write_text('window.LSTM_DATA = '+json.dumps(payload,ensure_ascii=False,allow_nan=False)+';',encoding='utf-8')
    summary='\n'.join(f"| {name} | {s['MAE']:.3f} | {s['RMSE']:.3f} | {s['R2']:.4f} |" for name,s in comparison.items())
    report=f'''# LSTM：北京官园站下一小时气温预测

输入过去24小时已知气温序列、窗口内温差及已知小时周期，预测下一小时温度变化，再加到前1小时温度上。没有把目标时刻的温度或天气作为输入。
结构：单层LSTM，4个序列输入，32个隐藏单元，线性输出1个温差。不是把线性回归改名成LSTM。

前80%历史是训练期；训练期内部按时间分出末20%作为验证，归一化统计只来自更早的拟合段。最多{max_epochs}轮、连续{patience}轮无改进早停，验证选{best_epoch}轮，再在完整训练期重训{best_epoch}轮。测试期不参与选轮数。
测试时每小时使用此前已发生的真实记录，模型参数固定；不是一次预测未来数月。缺测仅向前沿用最多3小时，窗口仍缺输入就不预测；目标缺测不填补。

| 方法 | MAE（℃） | RMSE（℃） | R² |
|---|---:|---:|---:|
{summary}

三种方法严格使用同一批{int(fair.sum())}个可评价时刻；整个测试期有{len(result)}小时。缺少完整窗口可能导致与旧版6993小时的评价数不同，因此不能混用旧分数。
最高R²不等于准确率，也不保证LSTM优于简单模型。是否改善以本表实际结果为准。该历史测试期之前已展示过，不代表全新外部验证。

训练与验证曲线：training_history.csv；重训曲线：final_training_history.csv；逐小时预测：test_predictions.csv；模型：lstm_checkpoint.pt及model.json。
运行环境：PyTorch {torch.__version__}，CPU；随机种子42。耗时{metrics['elapsed_seconds']:.1f}秒。
数据来源：https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data ，CC BY 4.0。
LSTM实现参考：https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
'''
    (OUT/'运行结果.md').write_text(report,encoding='utf-8');print(report,flush=True)
    from integrate_lstm import integrate
    integrate()


def draw_plots(result,history):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    def save(name):plt.tight_layout();plt.savefig(OUT/name,dpi=160);plt.close()
    plt.figure(figsize=(11,4.4));plt.plot(history.epoch,history.train_mse_standardized_delta,label='训练MSE')
    plt.plot(history.epoch,history.validation_mse_standardized_delta,label='验证MSE');plt.xlabel('轮数');plt.ylabel('标准化温差MSE');plt.title('LSTM训练与验证：只用训练期内部数据');plt.legend();save('01_训练曲线.png')
    week=result.iloc[:168];plt.figure(figsize=(12,4.6))
    for col,name,color in [('actual','实际气温','#263c56'),('lstm_prediction','LSTM','#14786b'),('linear_prediction','线性回归','#8c77af'),('persistence','沿用前1小时','#cba56b')]:
        plt.plot(week.timestamp,week[col],label=name,color=color,alpha=.85)
    plt.title('测试期最早7天：相同时间范围的模型对比');plt.ylabel('气温（℃）');plt.legend();save('02_一周模型对比.png')
    fair=result[result.fair_evaluation];plt.figure(figsize=(11,4.4))
    plt.hist(fair.actual-fair.lstm_prediction,bins=60,color='#14786b',alpha=.8);plt.xlabel('实际－LSTM预测（℃）');plt.ylabel('小时数');plt.title('LSTM测试期误差分布');save('03_误差分布.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=30);parser.add_argument('--patience',type=int,default=6)
    args=parser.parse_args()
    if args.epochs<1 or args.patience<1: parser.error('轮数和耐心值必须为正数。')
    train(args.epochs,args.patience)
