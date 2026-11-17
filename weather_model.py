"""官园站下一小时气温预测：纯线性公式与训练期缺测填补值。"""
import math
from datetime import datetime

BASE=['temp_prev','temp_change','temp_24','hour_sin','hour_cos']
EXTRA=['pressure_prev','dew_prev','wind_prev','rain_prev']
NAMES={'temp_prev':'前1小时气温（℃）','temp_change':'最近1小时气温变化（℃）',
       'temp_24':'前24小时气温（℃）','hour_sin':'目标小时正弦','hour_cos':'目标小时余弦',
       'pressure_prev':'前1小时气压（hPa）','dew_prev':'前1小时露点（℃）',
       'wind_prev':'前1小时风速（m/s）','rain_prev':'前1小时降水（mm）'}


def predict(model,values):
    result=model['intercept']
    for i,key in enumerate(model['features']):
        value=values.get(key)
        if value is None:
            value=model['medians'][key]
        if not math.isfinite(value): raise ValueError('输入包含非有限数值。')
        result+=value*model['coefficients'][i]
    return result


def scenario(timestamp,current,previous,same_hour_yesterday,pressure,dew,wind,rain):
    when=datetime.fromisoformat(timestamp)
    if when.minute or when.second: raise ValueError('请选择整点时刻。')
    if not all(math.isfinite(v) for v in [current,previous,same_hour_yesterday,pressure,dew,wind,rain]):
        raise ValueError('请输入有限数值。')
    if not all(-60<=v<=60 for v in [current,previous,same_hour_yesterday,dew]):
        raise ValueError('气温或露点超出本演示允许范围。')
    if not 800<=pressure<=1100 or wind<0 or rain<0: raise ValueError('气压、风速或降水范围有误。')
    return dict(temp_prev=current,temp_change=current-previous,temp_24=same_hour_yesterday,
        hour_sin=math.sin(2*math.pi*when.hour/24),hour_cos=math.cos(2*math.pi*when.hour/24),
        pressure_prev=pressure,dew_prev=dew,wind_prev=wind,rain_prev=rain)
