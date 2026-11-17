const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
process.chdir(__dirname);
const ids=[...fs.readFileSync('index.html','utf8').matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
const elements=Object.fromEntries(ids.map(id=>[id,{value:'',textContent:'',innerHTML:'',validity:{valid:true},classList:{toggle(){}},addEventListener(){}}]));
const context=vm.createContext({console,window:{scrollTo(){}},document:{getElementById:id=>elements[id],querySelectorAll:()=>[]}});
vm.runInContext(fs.readFileSync('outputs/demo-data.js','utf8'),context);
vm.runInContext(fs.readFileSync('app.js','utf8'),context);
vm.runInContext(`
for(const row of data.predictions){if(row.prediction!==null){
 const input=byTime.get(row.timestamp);
 if(Math.abs(estimate(input)-row.prediction)>1e-8)throw Error('Prediction mismatch '+row.timestamp);
}}
`,context);
assert.ok(Number.isFinite(Number(elements.estimate.textContent)));
for(let i=0;i<4;i++){vm.runInContext(`showPlot(${i})`,context);assert.ok(fs.existsSync(elements.plot.src));}
elements.range.value='all';vm.runInContext('drawChart()',context);
assert.ok(elements.chart.innerHTML.includes('7013小时'));
elements.when.value='2026-11-17T18:00';vm.runInContext('update()',context);
assert.equal(elements.estimate.textContent,'—');
console.log('PASS: browser predictions match all available Python forecasts; date validation and chart assets checked.');
