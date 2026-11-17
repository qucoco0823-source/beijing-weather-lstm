'use strict';
// 描述统计直接来自网页载入的原始天气记录，不使用填补后的预测输入。
const $=id=>document.getElementById(id),data=window.BEIJING_DATA;
if(!data||!window.LSTM_DATA)throw Error('缺少分析或模型数据，请先完成训练。');
const records=data.records,audit=data.metrics.audit;
const valid=v=>v!==null&&Number.isFinite(v);
const mean=values=>values.reduce((a,b)=>a+b,0)/values.length;
const display=v=>valid(v)?v.toFixed(1):'NA';
const timeText=s=>s.replace('T',' ');
document.querySelectorAll('[data-screen]').forEach(button=>button.addEventListener('click',()=>{
 document.querySelectorAll('.screen').forEach(section=>section.classList.toggle('active',section.id===button.dataset.screen));
 document.querySelectorAll('[data-screen]').forEach(item=>item.classList.toggle('active',item===button));window.scrollTo(0,0);
}));
$('record-count').textContent=records.length.toLocaleString('en-US')+'条';
$('duplicate-count').textContent=audit['重复时刻']+'条';
$('gap-count').textContent=audit['小时缺口']+'条';
const outside=records.filter(r=>(valid(r.TEMP)&&(r.TEMP< -60||r.TEMP>60))||(valid(r.PRES)&&(r.PRES<800||r.PRES>1100))||(valid(r.RAIN)&&r.RAIN<0)||(valid(r.WSPM)&&r.WSPM<0));
$('range-check').textContent=outside.length?'需核查'+outside.length+'条':'未发现超界值';
$('sample-rows').innerHTML=records.slice(0,5).map(r=>`<tr><td>${timeText(r.timestamp)}</td>${['TEMP','PRES','DEWP','RAIN','WSPM'].map(k=>`<td>${display(r[k])}</td>`).join('')}</tr>`).join('');
const names={TEMP:'气温 TEMP',PRES:'气压 PRES',DEWP:'露点 DEWP',RAIN:'降水 RAIN',WSPM:'风速 WSPM',wd:'风向 wd'};
$('missing-table').innerHTML=Object.entries(audit['缺失数量']).map(([key,count])=>`<tr><td>${names[key]}</td><td>${count}条</td><td>${(count/records.length*100).toFixed(3)}%</td><td>${key==='TEMP'?'分析时跳过NA；历史输入限3小时向前沿用；真实目标不填补':'保留NA；按有效观测统计；不进入最终LSTM'}</td></tr>`).join('');
let lastValue=null,run=0;
const filled=records.map(r=>{if(valid(r.TEMP)){lastValue=r.TEMP;run=0;return r.TEMP;}run++;return run<=3?lastValue:null;});
const firstMissing=records.findIndex(r=>!valid(r.TEMP));let endMissing=firstMissing;
while(endMissing<records.length&&!valid(records[endMissing].TEMP))endMissing++;
$('clean-example').innerHTML=records.slice(Math.max(0,firstMissing-1),endMissing+1).map(r=>{
 const i=records.indexOf(r);return `<tr><td>${timeText(r.timestamp)}</td><td>${display(r.TEMP)}</td><td>${display(filled[i])}${!valid(r.TEMP)&&valid(filled[i])?'（沿用）':''}</td><td>${valid(r.TEMP)?'保留实际值':'不填补，不评价'}</td></tr>`;
}).join('');
const lm=window.LSTM_DATA.metrics;
$('clean-result').textContent=`结果：${records.length.toLocaleString('en-US')}条原始记录完整保留。测试期${lm.test_hours}个小时中，${lm.evaluated_hours}个时刻可同时评价三种方法，${lm.excluded_hours}个时刻因目标或历史输入不足不评价。`;

function groupMeans(key,count,getGroup){
 const groups=Array.from({length:count},()=>[]);
 records.forEach(r=>{if(valid(r[key]))groups[getGroup(r)].push(r[key]);});
 return groups.map((values,i)=>({label:i,value:mean(values),count:values.length}));
}
// SVG保留明确的单位、坐标和样本量；不对真实曲线做平滑。
function statisticalChart(id,rows,{unit,label,bars=false}){
 const W=1080,H=300,L=65,R=22,T=35,B=50;
 const numbers=rows.map(r=>r.value),lower=bars?0:Math.floor(Math.min(...numbers)/5)*5-5,upper=Math.ceil(Math.max(...numbers)/5)*5+5;
 const x=i=>L+(i+.5)/rows.length*(W-L-R),y=v=>H-B-(v-lower)/(upper-lower)*(H-T-B);
 let parts=`<text x="${L}" y="18" fill="#5e6c72" font-size="13">${unit}</text>`;
 for(let tick=0;tick<=4;tick++){const v=lower+(upper-lower)*tick/4;parts+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#e3e8e9"/><text x="${L-10}" y="${y(v)+4}" text-anchor="end" font-size="12" fill="#5e6c72">${v.toFixed(1)}</text>`;}
 if(!bars)parts+=`<polyline fill="none" stroke="#176e67" stroke-width="2.5" points="${rows.map((r,i)=>`${x(i)},${y(r.value)}`).join(' ')}"/>`;
 rows.forEach((r,i)=>{const tooltip=`${label(i)}：${r.value.toFixed(2)}${unit}；有效观测${r.count}条`;
  parts+=bars?`<rect x="${x(i)-18}" y="${y(r.value)}" width="36" height="${y(0)-y(r.value)}" rx="2" fill="#478a91"><title>${tooltip}</title></rect>`:`<circle cx="${x(i)}" cy="${y(r.value)}" r="4" fill="#176e67"><title>${tooltip}</title></circle>`;
  if(rows.length<=12||i%3===0)parts+=`<text x="${x(i)}" y="${H-22}" text-anchor="middle" font-size="12" fill="#5e6c72">${label(i)}</text>`;
 });
 $(id).innerHTML=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${id==='monthly-chart'?'图1 各月份平均气温':id==='hourly-chart'?'图2 各时刻平均气温':'图3 各月份平均累计观测降水'}">${parts}</svg>`;
}
const monthly=groupMeans('TEMP',12,r=>+r.timestamp.slice(5,7)-1),hourly=groupMeans('TEMP',24,r=>+r.timestamp.slice(11,13));
statisticalChart('monthly-chart',monthly,{unit:'℃',label:i=>(i+1)+'月'});
statisticalChart('hourly-chart',hourly,{unit:'℃',label:i=>i+'时'});
const extrema=rows=>({low:rows.reduce((a,b)=>a.value<b.value?a:b),high:rows.reduce((a,b)=>a.value>b.value?a:b)});
const months=extrema(monthly),hours=extrema(hourly),temps=records.map(r=>r.TEMP).filter(valid);
$('monthly-finding').textContent=`四年合并统计中，${months.high.label+1}月平均气温最高（${months.high.value.toFixed(1)}℃），${months.low.label+1}月最低（${months.low.value.toFixed(1)}℃），月均差为${(months.high.value-months.low.value).toFixed(1)}℃。`;
$('temp-summary').textContent=`有效气温${temps.length.toLocaleString('en-US')}条；总体均温${mean(temps).toFixed(1)}℃；最低${Math.min(...temps).toFixed(1)}℃，最高${Math.max(...temps).toFixed(1)}℃。极值保留原始观测。`;
$('hourly-finding').textContent=`全期按小时分组，${hours.low.label}时均温最低，${hours.high.label}时最高，两者相差${(hours.high.value-hours.low.value).toFixed(1)}℃。这是四年汇总规律，不能解释为每一天都在同一时刻达到峰值。`;
const rainGroups=new Map();
records.forEach(r=>{const key=r.timestamp.slice(0,7);if(!rainGroups.has(key))rainGroups.set(key,{sum:0,count:0});if(valid(r.RAIN)){rainGroups.get(key).sum+=r.RAIN;rainGroups.get(key).count++;}});
const rainMonthly=Array.from({length:12},(_,i)=>{const values=[...rainGroups].filter(([key])=>+key.slice(5)===i+1).map(([,value])=>value);return {label:i,value:mean(values.map(v=>v.sum)),count:values.reduce((s,v)=>s+v.count,0)};});
statisticalChart('rain-chart',rainMonthly,{unit:'mm',label:i=>(i+1)+'月',bars:true});
const rainPeak=extrema(rainMonthly).high,rainSum=rainMonthly.reduce((s,r)=>s+r.value,0),summer=rainMonthly.slice(5,8).reduce((s,r)=>s+r.value,0);
$('rain-finding').textContent=`四年平均观测累计降水在${rainPeak.label+1}月最高（${rainPeak.value.toFixed(1)}mm）；6—8月占全年观测累计降水的${(summer/rainSum*100).toFixed(1)}%。`;
$('rain-note').textContent=`降水有效观测${records.filter(r=>valid(r.RAIN)).length.toLocaleString('en-US')}条，${audit['缺失数量'].RAIN}条缺测。这里统计已观测到的累计值，未估算缺测时段降水，累计值可能偏低。`;
const pairs=[];
for(let i=1;i<records.length;i++)if(valid(records[i].TEMP)&&valid(records[i-1].TEMP))pairs.push([records[i-1].TEMP,records[i].TEMP]);
const aMean=mean(pairs.map(r=>r[0])),bMean=mean(pairs.map(r=>r[1]));let cov=0,av=0,bv=0;
pairs.forEach(([a,b])=>{cov+=(a-aMean)*(b-bMean);av+=(a-aMean)**2;bv+=(b-bMean)**2;});
const lagCorrelation=cov/Math.sqrt(av*bv);
$('continuity-evidence').textContent=`全期相邻气温相关系数${lagCorrelation.toFixed(3)}（${pairs.length}对有效观测）`;
window.COURSE_STATS={monthly,hourly,rainMonthly,lagCorrelation,filledCount:filled.filter(valid).length,firstMissing,rangeViolations:outside.length};
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{$('fullscreen').textContent='可按F11全屏';}});
