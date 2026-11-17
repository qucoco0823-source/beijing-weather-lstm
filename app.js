'use strict';
const $=id=>document.getElementById(id), data=window.BEIJING_DATA;
if(!data) throw Error('请先运行训练生成北京演示数据。');
const model=data.model,m=data.metrics;
const byTime=new Map(data.inputs.map(r=>[r.timestamp,r]));
document.querySelectorAll('[data-screen]').forEach(b=>b.addEventListener('click',()=>{
 document.querySelectorAll('.screen').forEach(el=>el.classList.toggle('active',el.id===b.dataset.screen));
 document.querySelectorAll('[data-screen]').forEach(el=>el.classList.toggle('active',el===b));window.scrollTo(0,0);
}));
$('mae').textContent=m.scores.forecast.MAE.toFixed(2);
$('baseline-mae').textContent=m.scores.persistence.MAE.toFixed(2);
$('improvement').textContent=m.improvement_percent.toFixed(1)+'%';
$('test-note').textContent=`${m.test_start.replace('T',' ')} — ${m.test_end.replace('T',' ')}；${m.test_hours}小时，实际评价${m.evaluated_hours}小时。`;
$('score-table').innerHTML=[['线性回归',m.scores.forecast],['沿用当前气温',m.scores.persistence]].map(([name,s])=>`<tr><td>${name}</td><td>${s.MAE.toFixed(3)}</td><td>${s.RMSE.toFixed(3)}</td><td>${s.R2.toFixed(4)}</td></tr>`).join('');
$('feature-note').textContent=`最终选用${model.features.length}个输入：过去气温、最近气温变化、前24小时气温和目标小时的两个周期变量。`;
$('audit').innerHTML=Object.entries(m.audit['缺失数量']).map(([k,v])=>`<p class="small">${k} 缺测：${v}条</p>`).join('')+`<p class="small">重复时刻0，小时缺口0；测试期${m.excluded_hours}小时因目标或过去气温缺测不评价。</p>`;
$('split-note').textContent=`训练期${m.train_hours}小时，实际拟合${m.fit_hours}行；内部验证3次，每次30天。模型系数在测试期内保持不变。`;
$('coefficients').innerHTML=`<p class="small">截距 ${model.intercept.toFixed(6)}</p>`+model.features.map((key,i)=>`<p class="small">${model.feature_names[i]}：${model.coefficients[i].toFixed(6)}</p>`).join('');

function estimate(values){return model.intercept+model.features.reduce((sum,k,i)=>sum+(values[k]===null||values[k]===undefined?model.medians[k]:values[k])*model.coefficients[i],0);}
function drawChart(){
 const rows=$('range').value==='all'?data.predictions:data.predictions.slice(0,168),W=1100,H=310,L=50,R=15,T=15,B=40;
 const values=rows.flatMap(r=>[r.actual,r.prediction,r.persistence]).filter(v=>v!==null&&Number.isFinite(v));
 const min=Math.floor(Math.min(...values)/5)*5,max=Math.ceil(Math.max(...values)/5)*5+5;
 const x=i=>L+i/(rows.length-1)*(W-L-R),y=v=>H-B-(v-min)/(max-min)*(H-T-B);
 let content='';
 for(let v=min;v<=max;v+=5)content+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e8eeea"/><text x="${L-10}" y="${y(v)+4}" fill="#687c77" font-size="11" text-anchor="end">${v}℃</text>`;
 for(const [key,color,extra]of[['persistence','#cba56b','stroke-dasharray="5 4" opacity=".8"'],['actual','#263c56',''],['prediction','#15967e','']]){
  let path='',connected=false;
  rows.forEach((r,i)=>{const v=r[key];if(v===null||!Number.isFinite(v)){connected=false;return;}path+=`${connected?'L':'M'}${x(i)},${y(v)} `;connected=true;});
  content+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="${rows.length>200?1:2}" ${extra}/>`;
 }
 for(let s=0;s<=4;s++){const i=Math.round(s*(rows.length-1)/4);content+=`<text x="${x(i)}" y="${H-12}" fill="#687c77" font-size="11" text-anchor="${s===0?'start':s===4?'end':'middle'}">${rows[i].timestamp.slice(5).replace('T',' ')}</text>`;}
 $('chart').innerHTML=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${rows.length}小时实际气温与预测对比，缺测保留断线">${content}</svg>`;
}
$('range').addEventListener('change',drawChart);drawChart();
const plots=['01_月度气温.png','02_季节变化.png','03_一周预测.png','04_预测误差.png'];
function showPlot(i){$('plot').src='outputs/'+plots[i];$('plot').alt=plots[i].slice(3,-4);document.querySelectorAll('[data-plot]').forEach(b=>b.classList.toggle('active',+b.dataset.plot===i));}
document.querySelectorAll('[data-plot]').forEach(b=>b.addEventListener('click',()=>showPlot(+b.dataset.plot)));showPlot(0);
let auxiliary={};
function update(){
 const time=$('when').value,fields=['current','previous','yesterday'];
 if(!time||!$('when').validity.valid||time<model.data_start||time>model.data_end||!fields.every(k=>$(k).value!==''&&Number.isFinite(+$(k).value)&&$(k).validity.valid)){
  $('error').textContent='请填写有效的历史整点时刻和气温。';$('estimate').textContent='—';$('naive').textContent='—';$('change').textContent='—';return;
 }
 const hour=new Date(time+'Z').getUTCHours(),current=+$('current').value;
 const values={...auxiliary,temp_prev:current,temp_change:current-(+$('previous').value),temp_24:+$('yesterday').value,
 hour_sin:Math.sin(2*Math.PI*hour/24),hour_cos:Math.cos(2*Math.PI*hour/24)};
 const value=estimate(values);
 $('error').textContent='';$('estimate').textContent=value.toFixed(2);$('naive').textContent=current.toFixed(2)+'℃';
 $('change').textContent=(value-current>=0?'+':'')+(value-current).toFixed(2)+'℃';
 $('prediction-note').textContent=time.replace('T',' ')+' · 下一小时的历史情景估计';
}
function load(){
 const row=byTime.get($('when').value);
 if(!row||row.temp_prev===null){$('error').textContent='此前气温不足，请手动输入已知条件。';$('current').value='';update();return;}
 auxiliary={...row};
 $('current').value=row.temp_prev;
 $('previous').value=row.temp_prev-(row.temp_change??model.medians.temp_change);
 $('yesterday').value=row.temp_24??model.medians.temp_24;
 $('input-note').textContent='仅载入目标时刻之前的气温；历史辅助输入如缺测，使用训练期填补值。';update();
}
$('when').value=m.test_start;
$('form').addEventListener('input',update);$('form').addEventListener('change',update);$('form').addEventListener('submit',e=>e.preventDefault());
$('when').addEventListener('change',load);$('load').addEventListener('click',load);
$('warming').addEventListener('click',()=>{$('previous').value=(+$('current').value)-1;update();});
$('cooling').addEventListener('click',()=>{$('previous').value=(+$('current').value)+1;update();});load();
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{$('fullscreen').textContent='可按 F11 全屏';}});
