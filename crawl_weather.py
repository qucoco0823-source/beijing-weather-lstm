"""逐月采集公开历史天气HTML，解析表格并保留网页与请求凭证。"""
import argparse
import calendar
import csv
import hashlib
import json
import re
import time
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, ProxyHandler, build_opener
from urllib.robotparser import RobotFileParser

ROOT=Path(__file__).resolve().parent
BASE='https://www.tianqihoubao.com'
AGENT='WeatherCourseProject/1.0 (educational, low-frequency public HTML collection)'


class WeatherTable(HTMLParser):
    """只读取weather-table里的单元格，不执行网页脚本。"""
    def __init__(self):
        super().__init__();self.active=False;self.cell=None;self.row=[];self.rows=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='table' and 'weather-table' in attrs.get('class','').split():self.active=True
        if self.active and tag=='tr':self.row=[]
        if self.active and tag=='td':self.cell=[]
    def handle_data(self,text):
        if self.active and self.cell is not None:self.cell.append(text)
    def handle_endtag(self,tag):
        if self.active and tag=='td' and self.cell is not None:
            self.row.append(re.sub(r'\s+',' ',''.join(self.cell)).strip());self.cell=None
        if self.active and tag=='tr' and self.row:self.rows.append(self.row)
        if self.active and tag=='table':self.active=False


def parse_month(html,month,url):
    parser=WeatherTable();parser.feed(html);rows=[]
    for cells in parser.rows:
        if not any(cells):continue  # 表格内的空分隔行，不是每日记录。
        if len(cells)!=4:raise ValueError(f'{month}表格列数发生变化：{len(cells)}')
        raw_date,weather,temp,wind=cells
        match=re.fullmatch(r'(\d{4})年(\d{1,2})月(\d{1,2})日',raw_date)
        if not match:raise ValueError(f'无法识别日期：{raw_date}')
        day=date(*map(int,match.groups())).isoformat()
        if day[:7].replace('-','')!=month:raise ValueError('网页月份与数据日期不一致')
        rows.append({'date_raw':raw_date,'weather_raw':weather,'temperature_raw':temp,'wind_raw':wind,'date':day,'source_url':url})
    expected=calendar.monthrange(int(month[:4]),int(month[4:]))[1]
    if len({r['date'] for r in rows})!=expected:
        raise ValueError(f'{month}应有{expected}个不同日期，实际{len({r["date"] for r in rows})}个；不能冒充完整月份。')
    return rows


def months(start,end):
    year,month=int(start[:4]),int(start[4:])
    while f'{year:04d}{month:02d}'<=end:
        yield f'{year:04d}{month:02d}'
        month+=1
        if month==13:year+=1;month=1


def crawl(start='202101',end='202512',proxy=None,delay=1.2,refresh=False):
    raw=ROOT/'data/raw_html';raw.mkdir(parents=True,exist_ok=True)
    opener=build_opener(ProxyHandler({'https':proxy,'http':proxy})) if proxy else build_opener()
    log_path=ROOT/'data/crawl_log.jsonl'
    def log(event):
        with log_path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(event,ensure_ascii=False)+'\n')
    def request(url):
        for attempt in range(3):
            try:
                with opener.open(Request(url,headers={'User-Agent':AGENT,'Accept':'text/html,text/plain,*/*','Accept-Language':'zh-CN,zh;q=0.9'}),timeout=20) as reply:
                    payload=reply.read();charset=reply.headers.get_content_charset() or 'utf-8'
                    return payload,payload.decode(charset),reply.status
            except HTTPError as error:
                if error.code in (401,403,429):raise RuntimeError(f'网站拒绝或限流（{error.code}），停止采集。') from error
                if error.code in (404,406):raise
                if attempt==2:raise
            except (URLError,TimeoutError):
                if attempt==2:raise
            time.sleep(2*(attempt+1))
    robots_url=BASE+'/robots.txt';rules=None
    try:
        payload,robots,status=request(robots_url);(raw/'robots.txt').write_bytes(payload)
        rules=RobotFileParser();rules.parse(robots.splitlines())
        log({'kind':'robots','url':robots_url,'status':status,'fetched_at_utc':datetime.now(timezone.utc).isoformat()})
    except HTTPError as error:
        if error.code not in (404,406):raise
        log({'kind':'robots','url':robots_url,'status':error.code,'note':'规则页未能取得，记录响应；仅低频串行请求公开月份页，遇到拒绝或限流则停止。'})
    rows=[];manifest=[]
    targets=list(months(start,end))
    for index,month in enumerate(targets,1):
        url=f'{BASE}/lishi/beijing/month/{month}.html';cache=raw/f'beijing_{month}.html'
        if rules is not None and not rules.can_fetch(AGENT,url):raise RuntimeError('robots.txt不允许采集目标路径。')
        if cache.exists() and not refresh:
            payload=cache.read_bytes();html=payload.decode('utf-8');status='cached'
        else:
            payload,html,status=request(url)
            # 统一UTF-8缓存，哈希对应落盘内容，可离线复核解析。
            payload=html.encode('utf-8');cache.write_bytes(payload)
            time.sleep(max(delay,1.0))
        parsed=parse_month(html,month,url);rows.extend(parsed)
        event={'kind':'month','month':month,'url':url,'status':status,'rows':len(parsed),'unique_days':len({r['date'] for r in parsed}),'duplicate_rows':len(parsed)-len({r['date'] for r in parsed}),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'fetched_at_utc':datetime.fromtimestamp(cache.stat().st_mtime,timezone.utc).isoformat()}
        manifest.append(event);log(event)
        print(f'[{index:02d}/{len(targets)}] {month}: {len(parsed)}天，累计{len(rows)}条，{status}',flush=True)
    output=ROOT/'data/crawled_raw.csv'
    with output.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'source_name':'天气后报历史天气网页','source_home':BASE+'/lishi/beijing.html','source_type':'网站声明数据源于城市当天的天气预报信息，非本站验证的气象实测','city':'北京','start':rows[0]['date'],'end':rows[-1]['date'],'pages':len(manifest),'raw_rows':len(rows),'method':'urllib请求公开月份HTML + HTMLParser提取weather-table','delay_seconds':max(delay,1.0),'manifest':manifest}
    (ROOT/'data/crawl_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'采集完成：{len(targets)}个月份网页，{len(rows)}条每日记录。',flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',default='202101');parser.add_argument('--end',default='202512');parser.add_argument('--proxy');parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args();crawl(args.start,args.end,args.proxy,refresh=args.refresh)
