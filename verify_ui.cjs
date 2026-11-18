const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto');
process.chdir(__dirname);
const html=fs.readFileSync('index.html','utf8'),ids=[...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]);assert.equal(ids.length,new Set(ids).size);
const elements=Object.fromEntries(ids.map(id=>[id,{value:'',textContent:'',innerHTML:'',handlers:{},classList:{toggle(){}},addEventListener(event,fn){this.handlers[event]=fn;}}]));
const context=vm.createContext({console,window:{scrollTo(){}},document:{getElementById:id=>{assert.ok(elements[id],`Missing element ${id}`);return elements[id];},querySelectorAll:()=>[]}});
for(const file of ['outputs/demo-data.js','app.js'])vm.runInContext(fs.readFileSync(file,'utf8'),context);
const d=context.window.CRAWLED_WEATHER;
assert.equal(d.source.pages,60);assert.equal(d.records.length,1826);assert.equal(d.audit.raw_rows-d.audit.duplicates_removed,d.records.length);assert.equal(d.audit.missing_dates.length,0);
assert.equal(d.analysis.weather_counts.reduce((sum,r)=>sum+r.days,0),d.records.length);
let checkedHashes=0;for(const page of d.source.manifest){const file=`data/raw_html/beijing_${page.month}.html`;if(fs.existsSync(file)){const cache=fs.readFileSync(file);assert.equal(crypto.createHash('sha256').update(cache).digest('hex'),page.sha256);checkedHashes++;}}
for(const row of d.predictions){for(const model of ['ridge','forest','lstm']){const predicted=context.window.predictCrawledWeather(row.date,model);assert.ok(Math.abs(predicted-row[model])<(model==='lstm'?1e-4:1e-6),`${row.date} ${model}: ${predicted} != ${row[model]}`);}}
const date=d.predictions[100].date,target=d.records.find(r=>r.date===date),before=['ridge','forest','lstm'].map(m=>context.window.predictCrawledWeather(date,m));
const original=target.high;target.high=1000;assert.deepEqual(['ridge','forest','lstm'].map(m=>context.window.predictCrawledWeather(date,m)),before);target.high=original;
elements['reveal-actual'].handlers.click();assert.ok(elements['replay-error'].textContent.includes('先运行'));
elements['run-predict'].handlers.click();assert.equal(elements['replay-actual'].textContent,'尚未揭晓');assert.ok(elements['replay-result'].textContent.endsWith('℃'));elements['reveal-actual'].handlers.click();assert.ok(elements['replay-actual'].textContent.includes('绝对误差'));
for(const model of ['ridge','forest','lstm','persistence']){const mae=d.predictions.reduce((s,r)=>s+Math.abs(r.actual-r[model]),0)/d.predictions.length;assert.ok(Math.abs(mae-d.metrics.scores[model].MAE)<1e-8);}
console.log(`PASS: ${checkedHashes}/${d.source.pages} locally available page hashes; ${d.records.length} unique classified dates; all ${d.predictions.length} predictions for three models; past-only inputs; fair MAE and reveal guard.`);
