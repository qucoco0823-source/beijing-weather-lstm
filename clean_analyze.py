"""清洗自主采集的日天气记录，分类并输出探索性统计。"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def weather_class(text):
    text=str(text)
    if '雨' in text and '雪' in text:return '雨雪混合'
    if '雪' in text:return '雪'
    if '雨' in text:return '雨'
    if any(word in text for word in ['沙','尘']):return '沙尘'
    if any(word in text for word in ['雾','霾']):return '雾霾'
    if '阴' in text:return '阴'
    if '多云' in text:return '多云'
    if '晴' in text:return '晴'
    return '其他'


def clean():
    original=pd.read_csv(ROOT/'data/crawled_raw.csv',keep_default_na=False)
    df=original.copy();df['date']=pd.to_datetime(df.date,errors='raise')
    duplicates=int(df.date.duplicated().sum())
    if duplicates and df.groupby('date')[['temperature_raw','weather_raw','wind_raw']].nunique().gt(1).any().any():raise ValueError('同日期存在冲突记录，需人工核查。')
    df=df.drop_duplicates('date').sort_values('date').reset_index(drop=True)
    def temperatures(text):
        pieces=str(text).replace('−','-').split('/')
        if len(pieces)!=2:return np.nan,np.nan
        values=[]
        for piece in pieces:
            match=re.fullmatch(r'\s*(-?\d+(?:\.\d+)?)\s*[℃°C]*\s*',piece)
            values.append(float(match[1]) if match else np.nan)
        return values
    values=[temperatures(text) for text in df.temperature_raw]
    df[['high','low']]=np.asarray(values)
    invalid=(~df.high.between(-50,55))|(~df.low.between(-60,50))|(df.high<df.low)
    invalid&=df.high.notna()&df.low.notna()
    # 可疑温度保留原文，派生字段设缺测，防止静默改写真实记录。
    df.loc[invalid,['high','low']]=np.nan
    df['weather_class']=df.weather_raw.map(weather_class)
    df['rain_flag']=df.weather_raw.str.contains('雨').astype(int)
    df['snow_flag']=df.weather_raw.str.contains('雪').astype(int)
    df['wind_day']=df.wind_raw.str.split('/').str[0].str.strip()
    def wind_level(text):
        levels=re.findall(r'\d+',str(text));return max(map(int,levels)) if levels else np.nan
    df['wind_level']=df.wind_day.map(wind_level)
    df['month']=df.date.dt.month
    df['season']=df.month.map({12:'冬',1:'冬',2:'冬',3:'春',4:'春',5:'春',6:'夏',7:'夏',8:'夏',9:'秋',10:'秋',11:'秋'})
    df['day_range']=df.high-df.low
    # 温度中值只是最高/最低的派生指标，不称为实测日均温。
    df['temperature_midpoint']=(df.high+df.low)/2
    calendar=pd.date_range(df.date.min(),df.date.max(),freq='D')
    missing_dates=calendar.difference(df.date).strftime('%Y-%m-%d').tolist()
    audit={'raw_rows':len(original),'clean_rows':len(df),'duplicates_removed':duplicates,'missing_dates':missing_dates,'invalid_temperature_rows':int(invalid.sum()),'missing':{key:int(df[key].isna().sum()) for key in ['high','low','wind_level']},'rules':['日期标准化，移除无冲突的重复日期','气温字符串拆分并转数值，保留原文','最高温不得低于最低温，超范围值进入缺测核查','天气类别按雨雪混合、雪、雨、沙尘、雾霾、阴、多云、晴优先级互斥分类','描述统计跳过缺测，预测输入和真实目标均不填补，缺少完整7天历史时跳过样本'],'examples':df[['date_raw','temperature_raw','high','low','weather_raw','weather_class','wind_day','wind_level']].head(3).to_dict('records')}
    df.to_csv(ROOT/'data/cleaned_weather.csv',index=False,encoding='utf-8-sig')
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/cleaning_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'清洗完成：{len(original)}条采集记录，{len(df)}条清洗记录，{len(missing_dates)}个缺失日期。',flush=True)
    return df,audit


def analyze(df):
    monthly=df.groupby('month').agg(high=('high','mean'),low=('low','mean'),day_range=('day_range','mean'),days=('date','count'))
    seasons=df.groupby('season').agg(high=('high','mean'),low=('low','mean'),day_range=('day_range','mean'),days=('date','count')).reindex(['春','夏','秋','冬'])
    counts=df.weather_class.value_counts()
    summary={'monthly':monthly.reset_index().to_dict('records'),'seasons':seasons.reset_index().to_dict('records'),'weather_counts':[{'category':str(k),'days':int(v),'percent':float(v/len(df)*100)} for k,v in counts.items()],'high_max':float(df.high.max()),'low_min':float(df.low.min()),'lag1_correlation':float(df.high.corr(df.high.shift(1))),'rain_days':int(df.rain_flag.sum()),'classification_note':'一日一互斥类别，按固定优先级归类；雨天标记另行统计，可与雪标记重叠。'}
    (ROOT/'outputs/analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    frame,_=clean();analyze(frame)
