const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');process.chdir(__dirname);
const ids=[...fs.readFileSync('index.html','utf8').matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
assert.equal(ids.length,new Set(ids).size,'Duplicate HTML IDs');
const elements=Object.fromEntries(ids.map(id=>[id,{value:'',textContent:'',innerHTML:'',validity:{valid:true},handlers:{},classList:{toggle(){}},addEventListener(name,fn){this.handlers[name]=fn;}}]));
const context=vm.createContext({console,window:{scrollTo(){}},document:{getElementById:id=>elements[id],querySelectorAll:()=>[]}});
for(const file of ['outputs/demo-data.js','app.js','outputs/lstm/demo-data.js','lstm-runtime.js','lstm-ui.js'])vm.runInContext(fs.readFileSync(file,'utf8'),context);
const lstm=context.window.LSTM_DATA;
const valid=lstm.predictions.filter(r=>r.lstm_prediction!==null);
for(let i=0;i<40;i++){
 const row=valid[Math.floor(i*(valid.length-1)/39)];
 const history=context.window.beijingLSTMSequence(row.timestamp);
 const predicted=context.window.runWeatherLSTM(lstm.model,history.temperatures,history.hours);
 assert.ok(Math.abs(predicted-row.lstm_prediction)<1e-4,`Mismatch ${row.timestamp}`);
}
assert.equal(elements['lstm-actual'].textContent,'尚未揭晓');
elements['lstm-reveal'].handlers.click();assert.ok(elements['lstm-error'].textContent);
elements['lstm-run'].handlers.click();assert.ok(elements['lstm-result'].textContent.endsWith('℃'));
assert.equal(elements['lstm-actual'].textContent,'尚未揭晓');
elements['lstm-reveal'].handlers.click();assert.ok(elements['lstm-actual'].textContent.includes('实际'));
const time=elements['lstm-when'].value;
const before=context.window.beijingLSTMSequence(time);
const row=context.window.BEIJING_DATA.records.find(r=>r.timestamp===time),original=row.TEMP;row.TEMP+=1000;
const after=context.window.beijingLSTMSequence(time);assert.deepEqual(after,before);row.TEMP=original;
assert.equal(elements.mae.textContent,lstm.metrics.scores.lstm.MAE.toFixed(2));
assert.ok(elements['score-table'].innerHTML.includes('LSTM'));
assert.ok(fs.existsSync('outputs/lstm/01_训练曲线.png'));
console.log('PASS: browser LSTM agrees on 40 distributed samples; prediction-before-reveal, past-only history and displayed metrics verified.');
