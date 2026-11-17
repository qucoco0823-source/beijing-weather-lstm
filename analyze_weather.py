"""从原始CSV独立复算展示中的描述统计，保存可核对的分析结果。"""
import json
from pathlib import Path
from main import read_data

ROOT=Path(__file__).resolve().parent


def analyze():
    df,audit=read_data()
    monthly=df.groupby(df.timestamp.dt.month).TEMP.agg(['mean','count'])
    hourly=df.groupby(df.timestamp.dt.hour).TEMP.agg(['mean','count'])
    # min_count=1防止把整月NA误算成零降水；原始缺测不估算。
    rain=df.groupby(df.timestamp.dt.to_period('M')).RAIN.sum(min_count=1)
    rain_monthly=rain.groupby(rain.index.month).mean()
    rain_counts=df.groupby(df.timestamp.dt.month).RAIN.count()
    filled=df.TEMP.ffill(limit=3)
    result={
        'rows':len(df),'audit':audit,
        'monthly':[{'value':float(r['mean']),'count':int(r['count'])} for _,r in monthly.iterrows()],
        'hourly':[{'value':float(r['mean']),'count':int(r['count'])} for _,r in hourly.iterrows()],
        'rainMonthly':[{'value':float(rain_monthly.loc[i]),'count':int(rain_counts.loc[i])} for i in range(1,13)],
        'lagCorrelation':float(df.TEMP.corr(df.TEMP.shift(1))),
        'filledCount':int(filled.notna().sum()),
        'firstMissing':int(df.index[df.TEMP.isna()][0]),
        'units':{'monthly':'℃','hourly':'℃','rainMonthly':'mm'},
        'rain_note':'同月份四年观测累计降水的均值；NA保留，累计值可能低估。',
    }
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/course-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'已复算月度/小时气温、降水和清洗统计：{len(df)}条原始记录。')
    return result


if __name__=='__main__':analyze()
