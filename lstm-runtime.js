// PyTorch LSTM的独立浏览器推理：i/f/g/o门顺序，24步，零初始状态。
window.runWeatherLSTM=function(saved,temperatures,hours){
 const n=saved.hidden_size,w=saved.weights,s=saved.scaler;
 if(temperatures.length!==saved.window||hours.length!==saved.window||temperatures.some(v=>!Number.isFinite(v)))throw Error('需要完整的24小时已知气温。');
 let hidden=new Float64Array(n),cell=new Float64Array(n);
 const sigmoid=v=>1/(1+Math.exp(-Math.max(-50,Math.min(50,v))));
 for(let t=0;t<saved.window;t++){
  // 输入与PyTorch训练一致，先按float32保存，再以双精度计算门函数。
  const input=[(temperatures[t]-s.temperature_mean)/s.temperature_std,
   (t?temperatures[t]-temperatures[t-1]:0)/s.delta_std,
   Math.sin(2*Math.PI*hours[t]/24),Math.cos(2*Math.PI*hours[t]/24)].map(v=>Math.fround(v));
  const gates=new Float64Array(4*n);
  for(let g=0;g<4*n;g++){
   let value=w['lstm.bias_ih_l0'][g]+w['lstm.bias_hh_l0'][g];
   for(let j=0;j<4;j++)value+=w['lstm.weight_ih_l0'][g][j]*input[j];
   for(let j=0;j<n;j++)value+=w['lstm.weight_hh_l0'][g][j]*hidden[j];
   gates[g]=value;
  }
  const nextHidden=new Float64Array(n);
  for(let j=0;j<n;j++){
   cell[j]=sigmoid(gates[n+j])*cell[j]+sigmoid(gates[j])*Math.tanh(gates[2*n+j]);
   nextHidden[j]=sigmoid(gates[3*n+j])*Math.tanh(cell[j]);
  }
  hidden=nextHidden;
 }
 let change=w['output.bias'][0];
 for(let j=0;j<n;j++)change+=w['output.weight'][0][j]*hidden[j];
 return temperatures.at(-1)+change*s.delta_std;
};
