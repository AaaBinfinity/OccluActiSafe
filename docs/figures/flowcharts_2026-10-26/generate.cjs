/* Regenerate static thesis flowcharts. No model inference or simulation runs here. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const modules = process.env.FLOWCHART_NODE_MODULES ||
  '/Users/binfinity/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { instance } = require(path.join(modules, '@viz-js/viz/dist/viz.cjs'));
const sharp = require(path.join(modules, 'sharp'));
const root = __dirname;
const spec = JSON.parse(fs.readFileSync(path.join(root, 'source/graphs.json'), 'utf8'));
const palette = {
  process: { fill: '#F4F7FB', stroke: '#71859E', shape: 'box' },
  input: { fill: '#EDF7F2', stroke: '#38775E', shape: 'parallelogram' },
  model: { fill: '#EAF1FC', stroke: '#3B66A2', shape: 'box' },
  risk: { fill: '#FFF3DC', stroke: '#A36C1A', shape: 'box' },
  decision: { fill: '#FFF3DC', stroke: '#A36C1A', shape: 'diamond' },
  control: { fill: '#EAF3F5', stroke: '#397381', shape: 'box' },
  fallback: { fill: '#FDEEEA', stroke: '#A35540', shape: 'box' },
  evaluation: { fill: '#F1EFF9', stroke: '#76628D', shape: 'box' },
  output: { fill: '#EAF1FC', stroke: '#3B66A2', shape: 'box' }
};
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const dotq = s => '"' + String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"')
  .replace(/\n/g, '\\n') + '"';
const attrs = o => Object.entries(o).map(([k,v]) => k + '=' + dotq(v)).join(',');
function nodeLine(n) {
  const p = palette[n.kind || 'process'];
  return n.id + ' [' + attrs({
    label: n.label, shape: p.shape,
    style: p.shape === 'box' ? 'rounded,filled' : 'filled',
    fillcolor: p.fill, color: p.stroke,
    width: n._width || '1.8',
    height: n._height || '0.66',
    margin: n.kind === 'decision' ? '0.14,0.08' : '0.20,0.14',
    penwidth: '1.2'
  }) + '];';
}
function dot(g) {
  const out = [
    'digraph ' + dotq(g.id) + ' {',
    'graph [' + attrs({ rankdir: g.direction, bgcolor: 'white', pad: '0.26',
      nodesep: '0.36', ranksep: '0.46', splines: 'spline',
      fontname: 'PingFang SC', fontsize: '13', outputorder: 'edgesfirst', newrank: 'true' }) + '];',
    'node [' + attrs({fontname:'PingFang SC',fontsize:'13',fontcolor:'#22354F'}) + '];',
    'edge [' + attrs({fontname:'PingFang SC',fontsize:'10.5',color:'#62758C',
      fontcolor:'#485D75',penwidth:'1.1',arrowsize:'0.65'}) + '];'
  ];
  for (const group of g.groups || []) {
    out.push('subgraph cluster_' + group.id + ' {');
    out.push('graph [' + attrs({label:group.label,style:'rounded,filled',
      fillcolor:group.fill,color:'#D5DFEA',penwidth:'1',margin:'18',
      fontname:'PingFang SC',fontcolor:'#36516F',fontsize:'15'}) + '];');
    out.push(...g.nodes.filter(n=>n.group===group.id).map(nodeLine));
    out.push('}');
  }
  out.push(...g.nodes.filter(n=>!n.group).map(nodeLine));
  for (const row of g.ranks || []) out.push('{rank=same;' + row.join(';') + ';}');
  for (const [a,b,label,kind] of g.edges) {
    const opt = {};
    if (label) opt.label = label;
    if (kind === 'aux') Object.assign(opt,{style:'dashed',color:'#8B879E',
      fontcolor:'#716B85',arrowsize:'0.55'});
    if (kind === 'alert') Object.assign(opt,{color:'#AB573F',fontcolor:'#9C4F3A'});
    if (kind === 'feedback') Object.assign(opt,{style:'dashed',constraint:'false',
      color:'#71859E',tailport:g.feedback_side||'e',headport:g.feedback_side||'e',weight:'0.1'});
    out.push(a + ' -> ' + b + (Object.keys(opt).length?' ['+attrs(opt)+']':'') + ';');
  }
  out.push('}');
  return out.join('\n') + '\n';
}
function mermaid(g) {
  const m = ['flowchart ' + (g.direction==='TB'?'TD':g.direction)];
  function declaration(n) {
    const label = n.label.replace(/\n/g,'<br/>').replace(/"/g,'&quot;');
    if(n.kind==='decision') return '  '+n.id+'{"'+label+'"}';
    if(n.kind==='output') return '  '+n.id+'(["'+label+'"])';
    if(n.kind==='input') return '  '+n.id+'[/"'+label+'"/]';
    return '  '+n.id+'["'+label+'"]';
  }
  for(const group of g.groups||[]) {
    m.push('  subgraph '+group.id+'["'+group.label+'"]');
    m.push(...g.nodes.filter(n=>n.group===group.id).map(declaration));
    m.push('  end');
  }
  m.push(...g.nodes.filter(n=>!n.group).map(declaration));
  for(const [a,b,label,kind] of g.edges) {
    m.push('  '+a+(kind==='aux'||kind==='feedback'?' -.->':' -->')+
      (label?'|'+label+'| ':' ')+b);
  }
  for(const [kind,p] of Object.entries(palette)) {
    m.push('  classDef '+kind+' fill:'+p.fill+',stroke:'+p.stroke+
      ',color:#22354F,stroke-width:1.2px');
    const ns=g.nodes.filter(n=>(n.kind||'process')===kind);
    if(ns.length) m.push('  class '+ns.map(n=>n.id).join(',')+' '+kind);
  }
  return m.join('\n')+'\n';
}
function frameSvg(raw,g,num) {
  const start = raw.indexOf('<svg');
  const clean = raw.slice(start);
  const v = clean.match(/viewBox="([\d.\- ]+)"/);
  if(!v) throw new Error('Missing viewBox');
  const [, , w,h] = v[1].split(/\s+/).map(Number);
  const width = Math.ceil(Math.max(540,w+24)), height=Math.ceil(h+122);
  const inner=clean.slice(clean.indexOf('>')+1,clean.lastIndexOf('</svg>'));
  const x=(width-w)/2;
  const body=[
    '<svg xmlns="http://www.w3.org/2000/svg" width="'+width+'" height="'+height+
      '" viewBox="0 0 '+width+' '+height+'" role="img" aria-labelledby="title desc">',
    '<title id="title">'+esc('图 '+num+' '+g.title)+'</title>',
    '<desc id="desc">'+esc(g.caption+' '+spec.status)+'</desc>',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    '<g font-family="PingFang SC, Microsoft YaHei, sans-serif">',
    '<text x="24" y="29" fill="#213651" font-size="20" font-weight="600">'+
      esc('图 '+num+'  '+g.title)+'</text>',
    '<text x="24" y="52" fill="#5A6E85" font-size="11.5">'+esc(g.subtitle)+'</text>',
    '<line x1="24" y1="67" x2="'+(width-24)+'" y2="67" stroke="#D9E2EC"/>',
    '</g>',
    '<g transform="translate('+x+',78)">'+inner+'</g>',
    '<g font-family="PingFang SC, Microsoft YaHei, sans-serif" font-size="10" fill="#657A90">',
    '<text x="24" y="'+(height-14)+'">拟议方案示意 · 技术路线 V1.1 · '+spec.date+'</text>',
    '</g></svg>'
  ].join('\n');
  return {svg:body,width,height};
}
function validateGraph(g) {
  const ids=new Set(g.nodes.map(n=>n.id));
  if(ids.size!==g.nodes.length) throw new Error('Duplicate IDs: '+g.id);
  for(const [a,b]of g.edges) if(!ids.has(a)||!ids.has(b)) throw new Error('Unknown edge node');
  const incident=new Set(g.edges.flatMap(e=>e.slice(0,2)));
  for(const id of ids) if(!incident.has(id)) throw new Error('Disconnected node '+id);
  for(const n of g.nodes.filter(n=>n.kind==='decision')) {
    if(g.edges.filter(e=>e[0]===n.id).length<2) throw new Error('Unlabeled decision branch '+n.id);
  }
  return ids;
}
function layoutChecks(json,g) {
  const objects=(json.objects||[]).filter(o=>o.pos&&o.width&&o.height&&o.name);
  const boxes=objects.map(o=>{
    const [x,y]=o.pos.split(',').map(Number);
    return {id:o.name,x,y,w:+o.width*72,h:+o.height*72};
  });
  const overlaps=[];
  for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){
    const a=boxes[i],b=boxes[j];
    if(Math.abs(a.x-b.x)<(a.w+b.w)/2-0.3 &&
      Math.abs(a.y-b.y)<(a.h+b.h)/2-0.3) overlaps.push([a.id,b.id]);
  }
  if(overlaps.length) throw new Error(g.id+' overlapping nodes '+JSON.stringify(overlaps));
  return {nodes_checked:boxes.length,node_overlaps:0};
}
async function main(){
  fs.mkdirSync(path.join(root,'images'),{recursive:true});
  const viz=await instance();
  // Graphviz WASM does not have the system CJK font metrics. Measure real text
  // with the same font used by sharp, then reserve node geometry before layout.
  for(const g of spec.graphs)for(const n of g.nodes){
    const lines=n.label.split('\n');
    const metrics=await Promise.all(lines.map(line=>sharp({
      text:{text:esc(line),font:'PingFang SC 13',dpi:72,rgba:true}
    }).metadata()));
    const width=Math.max(...metrics.map(m=>m.width));
    const lineHeight=Math.max(17,...metrics.map(m=>m.height));
    const kind=n.kind||'process';
    const factor=kind==='decision'?1.8:kind==='input'?1.28:1;
    n._width=(Math.max(130,width*factor+36)/72).toFixed(4);
    n._height=((lines.length*lineHeight+(kind==='decision'?34:22))/72).toFixed(4);
    n._measured_text_width=width;
  }
  const manifest={date:spec.date,source_versions:spec.source_versions,
    status:spec.status,renderer:'@viz-js/viz 3.25.0, Graphviz dot, sharp PNG',figures:[]};
  for(let i=0;i<spec.graphs.length;i++) {
    const g=spec.graphs[i];validateGraph(g);
    const src=dot(g), mm=mermaid(g);
    fs.writeFileSync(path.join(root,'source',g.id+'.dot'),src);
    fs.writeFileSync(path.join(root,'source',g.id+'.mmd'),mm);
    const raw=viz.renderString(src,{engine:'dot',format:'svg'});
    const json=JSON.parse(viz.renderString(src,{engine:'dot',format:'json'}));
    const layout=layoutChecks(json,g);
    const framed=frameSvg(raw,g,i+1);
    fs.writeFileSync(path.join(root,'images',g.id+'.svg'),framed.svg);
    const image=await sharp(Buffer.from(framed.svg),{density:180})
      .png().toFile(path.join(root,'images',g.id+'.png'));
    manifest.figures.push({id:g.id,title:g.title,nodes:g.nodes.length,
      edges:g.edges.length,width:framed.width,height:framed.height,
      png_width:image.width,png_height:image.height,...layout,
      source_sha256:crypto.createHash('sha256').update(src).digest('hex')});
    console.log(g.id+' '+framed.width+'x'+framed.height+' '+g.nodes.length+' nodes');
  }
  const cards=spec.graphs.map((g,i)=>{
    const figure=manifest.figures[i];
    return '<section id="'+g.id+'"><header><div class="number">'+String(i+1).padStart(2,'0')+
      '</div><div><h2>'+esc(g.title)+'</h2><p>'+esc(g.caption)+'</p></div></header>'+
      '<div class="actions"><a href="images/'+g.id+'.svg" download>下载 SVG 矢量图</a>'+
      '<a href="images/'+g.id+'.png" download>下载 PNG 图片</a>'+
      '<a href="source/'+g.id+'.mmd" download>Mermaid 源码</a>'+
      '<a href="source/'+g.id+'.dot" download>DOT 源码</a></div>'+
      '<div class="canvas" tabindex="0" aria-label="'+esc(g.title)+'，可左右滚动">'+
      '<img src="images/'+g.id+'.svg" style="width:'+Math.min(1100,Math.max(660,figure.width))+
      'px" alt="'+esc(g.title+'。'+g.caption)+'" loading="'+(i?'lazy':'eager')+'"></div>'+
      '<p class="source">依据：'+esc(g.sources.join('；'))+'。拟议方案，不表示已完成实验。</p></section>';
  }).join('\n');
  const nav=spec.graphs.map((g,i)=>'<a href="#'+g.id+'">'+(i+1)+' '+esc(g.title)+'</a>').join('');
  const css=[
    ':root{color-scheme:light;--ink:#213651;--muted:#526880;--line:#dae3ed}',
    '*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f7fb;color:var(--ink);font-family:"PingFang SC","Microsoft YaHei",sans-serif}',
    'main{max-width:1240px;margin:0 auto;padding:36px 24px 64px}.eyebrow{font-size:12px;color:#4f6985;letter-spacing:.08em}',
    'h1{font-size:30px;line-height:1.45;margin:8px 0 12px}p{line-height:1.8}h1+p{max-width:900px;color:var(--muted);font-size:14px}',
    'nav{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0 32px}a{color:#244f86;text-underline-offset:3px}',
    'nav a{padding:9px 13px;border:1px solid #c9d7e7;border-radius:5px;text-decoration:none;background:white;font-size:13px}',
    'section{background:white;border:1px solid var(--line);border-radius:8px;margin:24px 0;padding:24px;scroll-margin-top:16px}',
    'section header{display:flex;gap:15px;align-items:flex-start}h2{font-size:21px;margin:0 0 7px}header p{font-size:13px;color:var(--muted);margin:0;max-width:980px}',
    '.number{font-size:24px;font-weight:600;color:#406995;line-height:1.3;min-width:36px}.actions{display:flex;flex-wrap:wrap;gap:18px;margin:16px 0 18px;font-size:13px}',
    '.canvas{overflow:auto;background:white;border:1px solid #e7edf4;border-radius:4px}.canvas img{display:block;height:auto;margin:0 auto;max-width:none}',
    '.source{font-size:12px;color:#657a90;margin:12px 0 0}.note{border-left:3px solid #8aa7c5;padding-left:14px;color:#4d6480;font-size:13px}',
    'a:focus-visible,.canvas:focus-visible{outline:3px solid #547caf;outline-offset:3px}',
    '@media(max-width:640px){main{padding:22px 12px 36px}h1{font-size:23px}section{padding:16px 12px}h2{font-size:18px}.actions{gap:13px}nav a{font-size:12px}}',
    '@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}',
    '@media print{body{background:white}main{padding:0}nav,.actions{display:none}section{break-before:page;border:0;padding:0}.canvas{overflow:visible;border:0}.canvas img{max-width:100%;width:100%!important}}'
  ].join('\n');
  const html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'+
    '<meta name="viewport" content="width=device-width, initial-scale=1"><title>毕业设计流程图集</title>'+
    '<style>'+css+'</style></head><body><main><div class="eyebrow">THESIS / FLOWCHARTS · '+spec.date+'</div>'+
    '<h1>超视野车路协同主动避撞 · 流程图集</h1>'+
    '<p>6 张流程图对应现有研究方案，提供可编辑的 Mermaid / DOT 源码，以及适合论文和答辩材料的 SVG / PNG。图中参数为仿真起步设置，模型、控制和正式实验仍待实施。</p>'+
    '<p class="note">桌面直接阅读；手机可在图内左右滚动。SVG 可独立打开继续放大，中文标签保留为可检索文字。</p>'+
    '<nav aria-label="图集目录">'+nav+'</nav>'+cards+'</main></body></html>';
  fs.writeFileSync(path.join(root,'index.html'),html);
  const fence=String.fromCharCode(96).repeat(3);
  let md='# 毕业设计流程图集\n\n日期：'+spec.date+'  \n图集版本：V1.0  \n'+
    '依据：主方案 V0.9、详细技术路线 V1.1；本次仅将既有设计可视化。所有图为拟议研究流程，不表示正式系统或实验已经完成。\n\n'+
    '[打开离线图集](index.html)\n\n';
  for(let i=0;i<spec.graphs.length;i++){
    const g=spec.graphs[i], base='';
    md+='## 图 '+(i+1)+' '+g.title+'\n\n'+g.caption+'\n\n'+
      '![图 '+(i+1)+' '+g.title+']('+base+'images/'+g.id+'.png)\n\n'+
      '[SVG 矢量图]('+base+'images/'+g.id+'.svg) · [PNG 图片]('+base+'images/'+g.id+'.png) · '+
      '[Mermaid 源码]('+base+'source/'+g.id+'.mmd) · [DOT 源码]('+base+'source/'+g.id+'.dot)\n\n'+
      fence+'mermaid\n'+mermaid(g)+fence+'\n\n依据：'+g.sources.join('；')+'。\n\n';
  }
  md+='## 使用与再生成\n\n论文优先插入 SVG；不支持 SVG 的编辑器可使用 PNG。Mermaid 便于在 Markdown 中修改；DOT 是本次导出排版源码。两类源码来自同一 graphs.json 节点关系，布局可能随渲染器变化。\n\n'+
    '再生成：在安装 @viz-js/viz 与 sharp 的 Node.js 环境运行图集目录下 generate.cjs；可通过 FLOWCHART_NODE_MODULES 指定依赖目录。\n\n'+
    '本次仅检查图形源码、布局和导出，不重新运行 53 个算法/系统测试，也不产生预测或碰撞率结果。\n';
  fs.writeFileSync(path.join(root,'毕业设计流程图集.md'),md);
  fs.writeFileSync(path.join(root,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
  const readme='# 流程图导出文件\n\n日期：'+spec.date+'\n\n'+
    '- index.html：离线预览与下载入口。\n- images/：6 张 SVG 与 6 张 PNG。\n- source/graphs.json：节点与关系的统一可编辑源。\n- source/*.mmd：Mermaid；source/*.dot：Graphviz。\n- generate.cjs：再生成脚本；依赖 @viz-js/viz 与 sharp。\n- manifest.json：导出尺寸、节点/边数与布局检查。\n\n'+
    '这是既有毕业设计方案的可视化交付，不是已实现系统或已完成实验的证据。\n';
  fs.writeFileSync(path.join(root,'README.md'),readme);
  console.log(JSON.stringify({figures:manifest.figures.length,status:'rendered',root}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
