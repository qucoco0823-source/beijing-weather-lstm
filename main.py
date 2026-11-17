"""北京官园站：质量检查、时间验证、简单线性回归、图表和演示导出。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from weather_model import BASE,EXTRA,NAMES,predict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
SOURCE='https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data'


def read_data():
    path=next((ROOT/'data/original_stations').glob('PRSA_Data_Guanyuan_*.csv'))
    raw=pd.read_csv(path)
    df=raw[['year','month','day','hour','TEMP','PRES','DEWP','RAIN','WSPM','wd']].copy()
    df['timestamp']=pd.to_datetime(df[['year','month','day','hour']],errors='raise')
    for k in ['TEMP','PRES','DEWP','RAIN','WSPM']:
        df[k]=pd.to_numeric(df[k],errors='raise')
    if df['timestamp'].duplicated().any(): raise ValueError('存在重复时刻，需核查原始记录。')
    df=df.sort_values('timestamp').reset_index(drop=True)
    if not df.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all(): raise ValueError('小时不连续，不能按行生成历史输入。')
    invalid=(~df.TEMP.dropna().between(-60,60)).any() or (~df.PRES.dropna().between(800,1100)).any()
    invalid=invalid or (df.RAIN.dropna()<0).any() or (df.WSPM.dropna()<0).any()
    if invalid: raise ValueError('发现范围异常，需核查而非静默删除。')
    audit={'原始小时数':len(df),'重复时刻':0,'小时缺口':0,
           '缺失数量':{k:int(df[k].isna().sum()) for k in ['TEMP','PRES','DEWP','RAIN','WSPM','wd']},
           '处理':'不填补真实预测目标；输入先仅向前沿用最多3小时，再用每个训练子集的中位数填补剩余辅助输入；前1小时气温仍缺失时不预测。'}
    return df,audit


def make_features(df):
    # ffill仅看过去，限制连续沿用长度，禁止bfill或跨未来的线性插值。
    known=df[['TEMP','PRES','DEWP','RAIN','WSPM']].ffill(limit=3)
    x=pd.DataFrame(index=df.index)
    x['temp_prev']=known.TEMP.shift(1)
    x['temp_change']=known.TEMP.shift(1)-known.TEMP.shift(2)
    x['temp_24']=known.TEMP.shift(24)
    x['hour_sin']=np.sin(2*np.pi*df.hour/24)
    x['hour_cos']=np.cos(2*np.pi*df.hour/24)
    for name,raw in [('pressure_prev','PRES'),('dew_prev','DEWP'),('wind_prev','WSPM'),('rain_prev','RAIN')]:
        x[name]=known[raw].shift(1)
    return x


def fit(x,y,features):
    allowed=y.notna()&x.temp_prev.notna()&(x.index>=24)
    medians=x.loc[allowed,features].median()
    if medians.isna().any(): raise ValueError('训练子集中某项输入全部缺失。')
    model=LinearRegression().fit(x.loc[allowed,features].fillna(medians),y.loc[allowed])
    exported={'features':features,'feature_names':[NAMES[k] for k in features],
        'medians':medians.to_dict(),'intercept':float(model.intercept_),'coefficients':model.coef_.tolist(),
        'model_type':'linear_regression','horizon_hours':1,'station':'北京官园站','fit_rows':int(allowed.sum())}
    return model,exported


def forecast(fitted,x):
    model,saved=fitted
    result=pd.Series(np.nan,index=x.index)
    allowed=x.temp_prev.notna()
    result.loc[allowed]=model.predict(x.loc[allowed,saved['features']].fillna(saved['medians']))
    return result


def scores(actual,pred):
    mask=actual.notna()&pred.notna()
    return {'hours':int(mask.sum()),'MAE':float(mean_absolute_error(actual[mask],pred[mask])),
        'RMSE':float(np.sqrt(mean_squared_error(actual[mask],pred[mask]))),'R2':float(r2_score(actual[mask],pred[mask]))}


def choose(x,y,times):
    rows=[]
    for fold,(a,b) in enumerate(TimeSeriesSplit(n_splits=3,test_size=24*30).split(x),1):
        for name,features in [('temperature',BASE),('weather',BASE+EXTRA)]:
            fitted=fit(x.iloc[a],y.iloc[a],features)
            result=scores(y.iloc[b],forecast(fitted,x.iloc[b]))
            rows.append({'variant':name,'fold':fold,'fit_end':times.iloc[a].max().isoformat(timespec='minutes'),
                'validation_start':times.iloc[b].min().isoformat(timespec='minutes'),
                'validation_end':times.iloc[b].max().isoformat(timespec='minutes'),**result})
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'validation_results.csv',index=False,encoding='utf-8-sig')
    ranking=table.groupby('variant')['MAE'].mean().to_dict()
    selected=min(ranking,key=ranking.get)
    print('训练期内部验证 MAE（℃）：',ranking,'选择：',selected)
    return selected,ranking


def plots(df,result):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    def save(name):
        plt.tight_layout();plt.savefig(OUT/name,dpi=160);plt.close()
    monthly=df.set_index('timestamp').TEMP.resample('MS').mean()
    plt.figure(figsize=(11,4.3));plt.plot(monthly.index,monthly.values,color='#14786b')
    plt.title('北京官园站：逐月平均气温');plt.ylabel('气温（℃）');save('01_月度气温.png')
    grouped=df.groupby(df.timestamp.dt.month).TEMP.agg(['mean','min','max'])
    plt.figure(figsize=(11,4.3));plt.plot(grouped.index,grouped['mean'],marker='o',color='#14786b',label='月均温')
    plt.fill_between(grouped.index,grouped['min'],grouped['max'],alpha=.14,color='#14786b',label='历史最低至最高')
    plt.xticks(range(1,13));plt.xlabel('月份');plt.ylabel('气温（℃）');plt.title('季节变化：各月份气温分布范围');plt.legend();save('02_季节变化.png')
    week=result.iloc[:168]
    plt.figure(figsize=(12,4.5));plt.plot(week.timestamp,week.actual,label='实际气温',color='#263c56')
    plt.plot(week.timestamp,week.prediction,label='下一小时线性预测',color='#14786b',alpha=.85)
    plt.plot(week.timestamp,week.persistence,label='沿用前1小时气温',color='#cba56b',linestyle='--',alpha=.8)
    plt.title('测试期最早7天：逐小时的一步预测');plt.ylabel('气温（℃）');plt.legend();save('03_一周预测.png')
    plt.figure(figsize=(11,4.3));error=result.actual-result.prediction
    plt.scatter(result.prediction,error,s=5,alpha=.25,color='#14786b');plt.axhline(0,linestyle='--',color='#cba56b')
    plt.xlabel('预测气温（℃）');plt.ylabel('实际－预测（℃）');plt.title('测试期误差分布：保留极端偏差');save('04_预测误差.png')


def main():
    OUT.mkdir(exist_ok=True)
    df,audit=read_data();x=make_features(df)
    split=int(len(df)*.8)
    selected,validation=choose(x.iloc[:split],df.TEMP.iloc[:split],df.timestamp.iloc[:split])
    features=BASE if selected=='temperature' else BASE+EXTRA
    fitted=fit(x.iloc[:split],df.TEMP.iloc[:split],features)
    _,saved=fitted
    test=df.iloc[split:];xtest=x.iloc[split:]
    prediction=forecast(fitted,xtest)
    persistence=xtest.temp_prev
    # 使用同一评价掩码，不能给各个方法挑不同的日期。
    eligible=test.TEMP.notna()&prediction.notna()&persistence.notna()
    actual=test.TEMP.where(eligible)
    result=pd.DataFrame({'timestamp':test.timestamp,'actual':test.TEMP,'prediction':prediction,
        'persistence':persistence,'absolute_error':abs(test.TEMP-prediction),'evaluated':eligible})
    metrics={'station':'北京官园站','unit':'℃','audit':audit,'train_hours':split,'test_hours':len(test),
        'fit_hours':saved['fit_rows'],'evaluated_hours':int(eligible.sum()),'excluded_hours':int((~eligible).sum()),
        'scores':{'forecast':scores(actual,prediction),'persistence':scores(actual,persistence)},
        'validation_mean_mae':validation,'selected':selected,
        'test_start':test.timestamp.min().isoformat(timespec='minutes'),'test_end':test.timestamp.max().isoformat(timespec='minutes')}
    metrics['improvement_percent']=(1-metrics['scores']['forecast']['MAE']/metrics['scores']['persistence']['MAE'])*100
    saved.update(training_end=df.timestamp.iloc[split-1].isoformat(timespec='minutes'),
        data_start=df.timestamp.min().isoformat(timespec='minutes'),data_end=df.timestamp.max().isoformat(timespec='minutes'),source=SOURCE)
    (OUT/'model.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    result.to_csv(OUT/'test_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'feature':features,'说明':[NAMES[k] for k in features],'coefficient':saved['coefficients']}).to_csv(OUT/'coefficients.csv',index=False,encoding='utf-8-sig')
    plots(df,result)
    records=df[['timestamp','TEMP','PRES','DEWP','RAIN','WSPM']].copy()
    records['timestamp']=records.timestamp.dt.strftime('%Y-%m-%dT%H:%M')
    result['timestamp']=result.timestamp.dt.strftime('%Y-%m-%dT%H:%M')
    input_rows=x.copy();input_rows['timestamp']=df.timestamp.dt.strftime('%Y-%m-%dT%H:%M')
    payload={'model':saved,'metrics':metrics,'records':json.loads(records.to_json(orient='records')),
        'inputs':json.loads(input_rows.to_json(orient='records',double_precision=14)),
        'predictions':json.loads(result.to_json(orient='records',double_precision=14))}
    (OUT/'demo-data.js').write_text('window.BEIJING_DATA = '+json.dumps(payload,ensure_ascii=False,allow_nan=False)+';',encoding='utf-8')
    for idx,p in prediction.dropna().items():
        mapping={k:None if pd.isna(v) else float(v) for k,v in x.loc[idx].items()}
        assert abs(predict(saved,mapping)-p)<1e-8
    s=metrics['scores'];report=f'''# 北京官园站：下一小时气温预测

来源：{SOURCE}；Chen, S. (2017). Beijing Multi-Site Air Quality，DOI:10.24432/C5RK5G，CC BY 4.0。
使用真实逐小时气象记录，未使用污染物字段。题目建议《基于Python的北京市天气变化分析与短期气温预测》。

模型：普通多元线性回归；输入方案={selected}，无树模型或神经网络。
训练期：{df.timestamp.iloc[0]}至{df.timestamp.iloc[split-1]}，{split}小时，实际拟合{saved['fit_rows']}行。
测试期：{metrics['test_start']}至{metrics['test_end']}，{len(test)}小时，其中{int(eligible.sum())}小时可公平评价，{int((~eligible).sum())}小时因目标缺测或过去气温缺测不评价。

| 方法 | MAE（℃） | RMSE（℃） | R² |
|---|---:|---:|---:|
| 线性回归 | {s['forecast']['MAE']:.3f} | {s['forecast']['RMSE']:.3f} | {s['forecast']['R2']:.4f} |
| 沿用前1小时气温 | {s['persistence']['MAE']:.3f} | {s['persistence']['RMSE']:.3f} | {s['persistence']['R2']:.4f} |

相比简单对照，MAE降低{metrics['improvement_percent']:.1f}%。R²不是准确率，短期气温本身有连续性，所以简单对照也很强，不能只展示一个很高的R²。

缺测检查：{json.dumps(audit,ensure_ascii=False)}
训练期内部3次滚动验证、每次30天，以MAE选择输入方案；各次填补统计量只来自当次训练子集。最终测试期不用于选择方案。
验证期平均MAE：{json.dumps(validation,ensure_ascii=False)}。

时刻D的所有天气输入最多来自D-1；只有目标小时日历可以提前知道。这里是逐小时一步预测，不是一次预测未来几个月。真实目标不填补、不平滑；缺测点在图中保留间断，不删掉难预测的峰谷。
温度变化、过去同小时气温和日内周期提供简单可解释的信息；不能保证突发降温等极端变化都准确。数据是2013—2017年北京官园站历史记录，不是实时天气预报服务。
'''
    (OUT/'运行结果.md').write_text(report,encoding='utf-8');print(report)
    print('北京训练完成：模型、4张图、全部测试明细和页面数据已更新。')


if __name__=='__main__': main()
