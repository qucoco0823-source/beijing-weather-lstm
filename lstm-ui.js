(function(){
'use strict';
const d=window.LSTM_DATA,base=window.BEIJING_DATA,$=id=>document.getElementById(id);
if(!d){$('lstm-status').textContent='LSTM训练尚未完成，请完成训练后刷新。';return;}
const m=d.metrics,model=d.model,score=m.scores;
$('mae').textContent=score.lstm.MAE.toFixed(2);
$('improvement').textContent=((1-score.lstm.MAE/score.persistence.MAE)*100).toFixed(1)+'%';
$('baseline-mae').textContent=score.persistence.MAE.toFixed(2);
$('test-note').textContent=`${m.test_start.replace('T',' ')} — ${m.test_end.replace('T',' ')}；${m.evaluated_hours}个相同有效时刻比较三种方法。`;
$('score-table').innerHTML=[['LSTM',score.lstm],['线性回归',score.linear],['沿用当前气温',score.persistence]].map(([name,s])=>`<tr><td>${name}</td><td>${s.MAE.toFixed(3)}</td><td>${s.RMSE.toFixed(3)}</td><td>${s.R2.toFixed(4)}</td></tr>`).join('');
$('feature-note').textContent=`LSTM读取过去${model.window}小时温度序列：1层、${model.hidden_size}个隐藏单元，预测下一小时温差。线性回归与沿用当前气温在同一批时刻作为对照。`;
$('lstm-status').textContent=`验证选择${m.selected_epochs}轮后，在完整训练期重训。相同测试样本下，${score.lstm.MAE<score.linear.MAE?'LSTM平均误差低于线性回归':'LSTM平均误差未优于线性回归'}；效果以实际数字为准。`;
$('audit').innerHTML+=`<p class="small">上面的基础缺测统计来自线性流程。LSTM需要完整24小时窗口，同条件对比实际评价${m.evaluated_hours}小时，${m.excluded_hours}小时因缺测不评价。</p>`;
$('lstm-split').textContent=`归一化统计只拟合到${model.scaler_fit_end.replace('T',' ')}。验证选轮数后，在截止${model.training_end.replace('T',' ')}的完整训练期重训；测试期不参与选轮数。`;
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
 if(!$('lstm-when').validity.valid||timestamp<base.model.data_start||timestamp>base.model.data_end){$('lstm-error').textContent='请选择数据范围内的历史整点时刻。';return;}
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
