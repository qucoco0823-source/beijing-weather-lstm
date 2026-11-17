(function(){
'use strict';
const d=window.LSTM_DATA,base=window.BEIJING_DATA,$=id=>document.getElementById(id);
if(!d){$('lstm-status').textContent='LSTM训练尚未完成，请完成训练后刷新。';return;}
const m=d.metrics,model=d.model,score=m.scores;
$('mae').textContent=score.lstm.MAE.toFixed(2);
$('test-note').textContent=`${m.test_start.replace('T',' ')} — ${m.test_end.replace('T',' ')}；${m.evaluated_hours}个相同有效时刻比较三种方法。`;
$('score-table').innerHTML=[['LSTM',score.lstm],['线性回归',score.linear],['沿用当前气温',score.persistence]].map(([name,s])=>`<tr><td>${name}</td><td>${s.MAE.toFixed(3)}</td><td>${s.RMSE.toFixed(3)}</td><td>${s.R2.toFixed(4)}</td></tr>`).join('');
$('lstm-split').textContent=`训练期：2013-03-01 00:00至${model.training_end.replace('T',' ')}；测试期：${m.test_start.replace('T',' ')}至${m.test_end.replace('T',' ')}。`;
$('lstm-status').textContent=`训练期内部按时间划分拟合与验证，验证选择${m.selected_epochs}轮后在完整训练期重训。标准化统计只使用截至${model.scaler_fit_end.replace('T',' ')}的早期拟合段；测试期不参与训练或选轮数。`;
const linearGain=(1-score.lstm.MAE/score.linear.MAE)*100,baselineGain=(1-score.lstm.MAE/score.persistence.MAE)*100;
$('score-finding').textContent=`LSTM平均误差为${score.lstm.MAE.toFixed(3)}℃，较线性回归降低${linearGain.toFixed(1)}%，较沿用上一小时气温降低${baselineGain.toFixed(1)}%。`;
const errors=d.predictions.filter(r=>r.fair_evaluation).map(r=>Math.abs(r.actual-r.lstm_prediction)).sort((a,b)=>a-b);
const percentile95=errors[Math.ceil(errors.length*.95)-1],maximum=errors.at(-1);
$('error-finding').textContent=`平均误差不代表每次误差：95%的有效测试时刻误差不超过${percentile95.toFixed(2)}℃，最大误差为${maximum.toFixed(2)}℃，突变时仍有较大偏差。`;
$('training-detail').textContent=`最多训练${m.max_epochs}轮，验证误差连续${m.patience}轮未改善则停止；本次选择${m.selected_epochs}轮。曲线来自训练期内部验证，未使用测试期选轮数。`;
$('project-conclusion').textContent=`本站气温呈现季节与日内变化，相邻小时具有较强连续性。在本次历史测试中，LSTM相较线性回归的MAE降低${linearGain.toFixed(1)}%，相较沿用上一小时降低${baselineGain.toFixed(1)}%；大误差时刻仍需保留并分析。`;
function drawComparison(){
 const rows=$('range').value==='all'?d.predictions:d.predictions.slice(0,168),W=1100,H=310,L=50,R=15,T=15,B=40;
 const numbers=rows.flatMap(r=>[r.actual,r.lstm_prediction,r.linear_prediction,r.persistence]).filter(v=>v!==null&&Number.isFinite(v));
 const min=Math.floor(Math.min(...numbers)/5)*5,max=Math.ceil(Math.max(...numbers)/5)*5+5;
 const x=i=>L+i/(rows.length-1)*(W-L-R),y=v=>H-B-(v-min)/(max-min)*(H-T-B);
 let content='';
 for(let v=min;v<=max;v+=5)content+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e8eeea"/><text x="${L-10}" y="${y(v)+4}" fill="#687c77" font-size="11" text-anchor="end">${v}℃</text>`;
 for(const [key,color,extra]of[['persistence','#cba56b','stroke-dasharray="5 4"'],['linear_prediction','#8c77af','opacity=".7"'],['actual','#263c56',''],['lstm_prediction','#15967e','']]){
  let path='',connected=false;rows.forEach((r,i)=>{const value=r[key];if(value===null||!Number.isFinite(value)){connected=false;return;}path+=`${connected?'L':'M'}${x(i)},${y(value)} `;connected=true;});
  content+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="${rows.length>200?1:2}" ${extra}/>`;
 }
 for(let s=0;s<=4;s++){const i=Math.round(s*(rows.length-1)/4);content+=`<text x="${x(i)}" y="${H-12}" fill="#687c77" font-size="11" text-anchor="${s===0?'start':s===4?'end':'middle'}">${rows[i].timestamp.slice(5).replace('T',' ')}</text>`;}
 $('chart').innerHTML=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${rows.length}小时LSTM、线性回归、简单对照和实际气温">${content}</svg>`;
}
$('range').addEventListener('change',drawComparison);drawComparison();
const lookup=new Map(base.records.map(r=>[r.timestamp,r]));
const iso=ms=>new Date(ms).toISOString().slice(0,16);
function knownTemperature(ms){
 for(let offset=0;offset<=3;offset++){
  const row=lookup.get(iso(ms-offset*3600000));
  if(row&&row.TEMP!==null&&Number.isFinite(row.TEMP))return row.TEMP;
 }
 throw Error('此前24小时中有无法沿用的缺测，不能预测。');
}
function sequenceAt(timestamp){
 const target=Date.parse(timestamp+'Z');if(!Number.isFinite(target))throw Error('日期无效。');
 const temperatures=[],hours=[];
 for(let ago=model.window;ago>=1;ago--){const ms=target-ago*3600000;temperatures.push(knownTemperature(ms));hours.push(new Date(ms).getUTCHours());}
 return {temperatures,hours};
}
window.beijingLSTMSequence=sequenceAt;
let loaded=null,computed=null;
function clearReplay(){loaded=null;computed=null;$('lstm-result').textContent='—';$('lstm-actual').textContent='尚未揭晓';$('lstm-error').textContent='';}
function loadSequence(){
 clearReplay();const timestamp=$('lstm-when').value;
 if(!$('lstm-when').validity.valid||timestamp<m.test_start||timestamp>m.test_end||!timestamp.endsWith(':00')){$('lstm-error').textContent='请选择测试期内的历史整点时刻。';return;}
 try{
  loaded={timestamp,...sequenceAt(timestamp)};
  $('lstm-history').textContent=`${iso(Date.parse(timestamp+'Z')-24*3600000).replace('T',' ')} 至 ${iso(Date.parse(timestamp+'Z')-3600000).replace('T',' ')}；共24个已知气温，最后为${loaded.temperatures.at(-1).toFixed(2)}℃。`;
  const low=Math.min(...loaded.temperatures)-1,high=Math.max(...loaded.temperatures)+1;
  const points=loaded.temperatures.map((v,i)=>`${20+i/23*660},${115-(v-low)/(high-low)*100}`).join(' ');
  $('lstm-history-chart').innerHTML=`<svg class="chart" viewBox="0 0 700 135" role="img" aria-label="仅显示此前24小时已知气温"><polyline fill="none" stroke="#14786b" stroke-width="2.5" points="${points}"/></svg>`;
  $('lstm-sample-type').textContent=timestamp>model.training_end?'测试期历史回放：不参与拟合或选轮数。':'训练期历史回放：可演示，但不代表测试成绩。';
 }catch(error){$('lstm-error').textContent=error.message;}
}
$('lstm-when').value=m.test_start;$('lstm-load').addEventListener('click',loadSequence);$('lstm-when').addEventListener('change',loadSequence);
$('lstm-run').addEventListener('click',()=>{
 if(!loaded||loaded.timestamp!==$('lstm-when').value)loadSequence();
 if(!loaded)return;
 computed=window.runWeatherLSTM(model,loaded.temperatures,loaded.hours);
 $('lstm-result').textContent=computed.toFixed(2)+'℃';$('lstm-actual').textContent='尚未揭晓';$('lstm-error').textContent='';
});
$('lstm-reveal').addEventListener('click',()=>{
 if(!loaded||computed===null){$('lstm-error').textContent='请先运行预测，再揭晓实际值。';return;}
 const target=lookup.get(loaded.timestamp);
 if(!target||target.TEMP===null){$('lstm-actual').textContent='真实气温缺测，不计算误差。';return;}
 $('lstm-actual').textContent=`实际${target.TEMP.toFixed(2)}℃；LSTM误差${Math.abs(target.TEMP-computed).toFixed(2)}℃；沿用前1小时误差${Math.abs(target.TEMP-loaded.temperatures.at(-1)).toFixed(2)}℃。`;
});
loadSequence();
})();
