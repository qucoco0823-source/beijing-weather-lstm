"""将LSTM结果加入原有本地界面，保留线性对照的手动演示。"""
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def integrate():
    path=ROOT/'index.html';html=path.read_text(encoding='utf-8')
    if 'lstm-ui.js' in html:return
    replacements={
        '<title>北京天气 · 下一小时气温预测</title>':'<title>北京天气 · LSTM气温预测与对比</title>',
        '北京官园站天气分析与短期气温预测 · 普通多元线性回归':'北京官园站天气分析与短期气温预测 · LSTM与简单模型同条件比较',
        '模型平均绝对误差':'LSTM平均绝对误差',
        '<span><i></i>线性预测</span>':'<span><i></i>LSTM</span><span><i style="background:#8c77af"></i>线性回归</span>',
        '简单模型，保留真实误差':'LSTM与简单方法同场比较',
        '没有加入树模型或神经网络，也没有平滑真实曲线、删除峰谷来改善图形。':'LSTM是实际训练的循环神经网络；保留线性回归和简单对照，不平滑真实曲线或删除难预测的峰谷。',
        'data-screen="method">代码与方法':'data-screen="method">LSTM方法',
    }
    for a,b in replacements.items():
        if a not in html:raise ValueError('页面结构与预期不符：'+a)
        html=html.replace(a,b)
    training='''<div class="panel"><h2>LSTM怎样训练？</h2><p id="lstm-status"></p><img class="plot-img" src="outputs/lstm/01_训练曲线.png" alt="LSTM训练和验证曲线"><p class="small">曲线仅来自训练期内部。验证确定轮数后，在完整训练期重训；最后测试期不参与早停或选轮数。</p></div>'''
    html=html.replace('</section>\n<section id="explore"',training+'</section>\n<section id="explore"',1)
    replay='''<div class="panel"><h2>LSTM：先预测，再揭晓实际值</h2><p>选择一个历史时刻，模型只读取此前24小时气温；真实值在点击揭晓后展示。</p><div class="field" style="max-width:360px"><label for="lstm-when">预测目标时刻</label><input id="lstm-when" type="datetime-local" step="3600" min="2013-03-02T00:00" max="2017-02-28T23:00" required></div><div class="scenario"><button type="button" id="lstm-load">载入此前24小时</button><button type="button" id="lstm-run">运行LSTM预测</button><button type="button" id="lstm-reveal">揭晓实际气温</button></div><p id="lstm-history" class="small"></p><div id="lstm-history-chart"></div><p id="lstm-sample-type" class="small"></p><div class="bottom-grid"><div class="result-card"><span class="badge">LSTM下一小时预测</span><div class="result-number" id="lstm-result">—</div></div><div class="panel"><h3>历史实际值</h3><p id="lstm-actual">尚未揭晓</p><p class="error" id="lstm-error" role="alert"></p></div></div></div>'''
    match=re.search(r'<section id="predict".*?</section>',html,re.S)
    if not match:raise ValueError('找不到预测区域。')
    section=match.group().replace('<div class="prediction-grid">',replay+'<details><summary>展开线性回归对照：手动修改三个气温输入</summary><div class="prediction-grid">',1)
    section=section[:-len('</section>')]+'</details></section>'
    section=section.replace('用已知气温，预测下一小时。','用过去24小时，让LSTM预测下一小时。')
    html=html[:match.start()]+section+html[match.end():]
    method='''<section id="method" class="screen"><div class="hero"><div><div class="eyebrow">03 / LSTM METHOD</div><h1>读取一段历史，学习温度如何变化。</h1><p>单层LSTM、32个隐藏单元、24小时窗口，输出下一小时温差。</p></div></div><div class="steps"><div class="step"><b>01</b><h3>过去24小时</h3><p class="small">窗口只到目标前1小时，整理气温、窗口内温差及已知小时周期。</p></div><div class="step"><b>02</b><h3>学习序列变化</h3><p class="small">LSTM的门控制保留或更新历史信息，输出最后一步的隐藏状态。</p></div><div class="step"><b>03</b><h3>估计下一小时温差</h3><p class="small">线性输出层给出温差，再加到前1小时气温上，得到预测温度。</p></div><div class="step"><b>04</b><h3>公平检验</h3><p class="small">LSTM、线性回归、沿用当前温度使用同一批有效测试时刻。</p></div></div><div class="panel"><h2>验证选轮数，测试只用于检验</h2><p id="lstm-split"></p><p class="small">标准化只使用拟合段统计量；缺测只向前沿用最多3小时，窗口仍缺则不预测。真实目标不填补。</p></div><div class="panel"><h3>如何讲清楚LSTM？</h3><p>“它不像线性回归只看几个整理后的数字，而是依次读取过去24小时。门控机制保留有用历史，最后估计下一小时会升温还是降温、变化多少。”</p><p class="small">单层网络保持适中规模。是否优于简单模型以真实测试误差为准；R²不是准确率，不代表能一次预测未来数月。</p><details><summary>线性回归对照模型系数</summary><div id="coefficients"></div></details><p id="split-note" class="small"></p></div></section>'''
    html=re.sub(r'<section id="method".*?</section>',method,html,flags=re.S)
    html=html.replace('<script src="app.js"></script>','<script src="app.js"></script><script src="outputs/lstm/demo-data.js"></script><script src="lstm-runtime.js"></script><script src="lstm-ui.js"></script>')
    path.write_text(html,encoding='utf-8')


if __name__=='__main__':integrate()
