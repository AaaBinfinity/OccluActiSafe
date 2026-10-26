/* Publication figures: explicit orthogonal layout, black/white, pt geometry.
   No inference, training or simulation is performed by this script. */
const fs=require('fs'), path=require('path'), crypto=require('crypto');
// Resolve installed packages normally; an explicit directory can override them.
function loadDependency(name) {
  const override = process.env.FLOWCHART_NODE_MODULES;
  try {
    const request = override ? require.resolve(name, { paths: [path.resolve(override)] }) : name;
    return require(request);
  } catch (error) {
    if (error.code !== 'MODULE_NOT_FOUND' && error.code !== 'ERR_MODULE_NOT_FOUND') throw error;
    console.error('无法加载流程图依赖 ' + name + '。请在仓库根目录运行：\n' +
      'npm install --no-save --package-lock=false sharp\n' +
      '也可将 FLOWCHART_NODE_MODULES 设为已有 node_modules 目录的绝对路径。');
    process.exit(1);
  }
}
const sharp = loadDependency('sharp');
const ROOT=__dirname;
const spec=JSON.parse(fs.readFileSync(path.join(ROOT,'source/figures.json'),'utf8'));
const FONT=10.5, LINE=0.75, LEADING=14;
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
  .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const mm=pt=>(pt*25.4/72).toFixed(5);
const f=s=>Number(s.toFixed(4));
function runs(text){
  return String(text).match(/[\x00-\x7F]+|[^\x00-\x7F]+/g)||[];
}
function pango(text){
  return runs(text).map(r=>'<span font_family="'+
    (/^[\x00-\x7F]+$/.test(r)?'Times New Roman':'Songti SC')+'">'+esc(r)+'</span>').join('');
}
function textSvg(text,x,y){
  const ts=runs(text).map(r=>'<tspan font-family="'+
    (/^[\x00-\x7F]+$/.test(r)?'Times New Roman, Times, serif':'Songti SC, SimSun, STSong, serif')+'">'+esc(r)+'</tspan>').join('');
  return '<text x="'+f(x)+'" y="'+f(y+FONT*0.34)+'" text-anchor="middle" font-size="'+FONT+
    '" fill="#000000">'+ts+'</text>';
}
function nodeSvg(n){
  let shape;
  if(n.shape==='decision'){
    shape='<polygon points="'+[[n.x+n.w/2,n.y],[n.x+n.w,n.y+n.h/2],
      [n.x+n.w/2,n.y+n.h],[n.x,n.y+n.h/2]].map(p=>p.join(',')).join(' ')+'"/>';
  }else if(n.shape==='terminal'){
    shape='<ellipse cx="'+(n.x+n.w/2)+'" cy="'+(n.y+n.h/2)+'" rx="'+n.w/2+'" ry="'+n.h/2+'"/>';
  }else{
    shape='<rect x="'+n.x+'" y="'+n.y+'" width="'+n.w+'" height="'+n.h+'"/>';
  }
  let out='<g id="node_'+n.id+'" fill="#ffffff" stroke="#000000" stroke-width="'+LINE+'">'+shape+'</g>';
  const cy=n.y+n.h/2;
  n.text.forEach((s,i)=>{out+=textSvg(s,n.x+n.w/2,cy+(i-(n.text.length-1)/2)*LEADING);});
  return out;
}
function edgeSvg(e){
  const d=e.points.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join(' ');
  return '<path d="'+d+'" fill="none" stroke="#000000" stroke-width="'+LINE+
    '" stroke-linecap="butt" stroke-linejoin="miter" marker-end="url(#arrow)"/>';
}
function drawSvg(g){
  const out=[
    '<svg xmlns="http://www.w3.org/2000/svg" xml:space="preserve" width="'+mm(g.width)+'mm" height="'+mm(g.height)+
      'mm" viewBox="0 0 '+g.width+' '+g.height+'" role="img" aria-labelledby="title desc">',
    '<title id="title">'+esc(g.title)+'</title><desc id="desc">'+esc(g.caption)+'</desc>',
    '<defs><marker id="arrow" markerUnits="userSpaceOnUse" markerWidth="6" markerHeight="4.5" '+
      'viewBox="0 0 6 4.5" refX="6" refY="2.25" orient="auto"><path d="M0 0 L6 2.25 L0 4.5 Z" fill="#000000"/></marker></defs>',
    '<rect x="0" y="0" width="'+g.width+'" height="'+g.height+'" fill="#ffffff"/>'
  ];
  for(const group of g.groups||[]){
    out.push('<rect x="'+group.x+'" y="'+group.y+'" width="'+group.w+'" height="'+group.h+
      '" fill="#ffffff" stroke="#000000" stroke-width="0.6" stroke-dasharray="4 3"/>');
    out.push(textSvg(group.title,group.x+group.w/2,group.y+15));
  }
  out.push(...g.edges.map(edgeSvg));
  out.push(...g.nodes.map(nodeSvg));
  for(const e of g.edges)if(e.label){
    out.push('<rect x="'+f(e.label_x-e._label_width/2-1.8)+'" y="'+f(e.label_y-FONT/2-1.2)+
      '" width="'+f(e._label_width+3.6)+'" height="'+(FONT+2.4)+'" fill="#ffffff"/>');
    out.push(textSvg(e.label,e.label_x,e.label_y));
  }
  out.push('</svg>');
  return out.join('\n')+'\n';
}
function segInBox(a,b,n){
  const eps=0.1, x1=n.x+eps,x2=n.x+n.w-eps,y1=n.y+eps,y2=n.y+n.h-eps;
  if(a[0]===b[0])return a[0]>x1&&a[0]<x2&&Math.max(Math.min(a[1],b[1]),y1)<Math.min(Math.max(a[1],b[1]),y2);
  return a[1]>y1&&a[1]<y2&&Math.max(Math.min(a[0],b[0]),x1)<Math.min(Math.max(a[0],b[0]),x2);
}
function validateLayout(g){
  const ids=new Set(g.nodes.map(n=>n.id));
  if(ids.size!==g.nodes.length)throw new Error('Duplicate node IDs');
  for(let i=0;i<g.nodes.length;i++){
    const a=g.nodes[i];
    if(a.x<0||a.y<0||a.x+a.w>g.width||a.y+a.h>g.height)throw new Error('Out of bounds '+a.id);
    for(const b of g.nodes.slice(i+1))
      if(Math.max(a.x,b.x)<Math.min(a.x+a.w,b.x+b.w) &&
         Math.max(a.y,b.y)<Math.min(a.y+a.h,b.y+b.h))throw new Error('Overlap '+a.id+' '+b.id);
  }
  for(const e of g.edges){
    if(!ids.has(e.from)||!ids.has(e.to))throw new Error('Bad edge node');
    for(let i=1;i<e.points.length;i++){
      const a=e.points[i-1],b=e.points[i];
      if(a[0]!==b[0]&&a[1]!==b[1])throw new Error('Non-orthogonal edge');
      for(const n of g.nodes)
        if(n.id!==e.from&&n.id!==e.to&&segInBox(a,b,n))throw new Error(g.id+' edge '+e.from+'>'+e.to+' through '+n.id);
    }
  }
}
function drawioPage(g,index){
  const scale=4/3, p=v=>f(v*scale), all=['<diagram id="fig'+index+'" name="'+esc(g.title)+'">'+
    '<mxGraphModel grid="1" gridSize="10" page="1" pageScale="1" pageWidth="'+p(g.width)+
    '" pageHeight="'+p(g.height)+'"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'];
  for(const [i,group]of (g.groups||[]).entries()){
    all.push('<mxCell id="group'+i+'" value="'+esc(group.title)+'" style="rounded=0;whiteSpace=wrap;html=1;'+
      'dashed=1;fillColor=#ffffff;strokeColor=#000000;strokeWidth=0.8;fontFamily=SimSun;fontSize=14;'+
      'verticalAlign=top;spacingTop=6;align=center;" vertex="1" parent="1"><mxGeometry x="'+p(group.x)+
      '" y="'+p(group.y)+'" width="'+p(group.w)+'" height="'+p(group.h)+'" as="geometry"/></mxCell>');
  }
  for(const n of g.nodes){
    const shape=n.shape==='decision'?'rhombus':n.shape==='terminal'?'ellipse':'rectangle';
    const value=n.text.map(t=>runs(t).map(r=>'<span style="white-space:pre;font-family:'+
      (/^[\x00-\x7F]+$/.test(r)?'Times New Roman':'SimSun')+'">'+esc(r)+'</span>').join('')).join('<br>');
    all.push('<mxCell id="'+n.id+'" value="'+esc(value)+'" style="shape='+shape+
      ';rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;'+
      'strokeWidth=1;fontColor=#000000;fontFamily=SimSun;fontSize=14;align=center;verticalAlign=middle;" '+
      'vertex="1" parent="1"><mxGeometry x="'+p(n.x)+'" y="'+p(n.y)+'" width="'+p(n.w)+
      '" height="'+p(n.h)+'" as="geometry"/></mxCell>');
  }
  for(const [i,e]of g.edges.entries()){
    const a=g.nodes.find(n=>n.id===e.from),b=g.nodes.find(n=>n.id===e.to);
    const first=e.points[0],last=e.points.at(-1);
    const style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#000000;strokeWidth=1;'+
      'endArrow=block;endFill=1;fontFamily=SimSun;fontSize=14;fontColor=#000000;'+
      'exitX='+f((first[0]-a.x)/a.w)+';exitY='+f((first[1]-a.y)/a.h)+';'+
      'entryX='+f((last[0]-b.x)/b.w)+';entryY='+f((last[1]-b.y)/b.h)+';';
    all.push('<mxCell id="edge'+i+'" value="'+esc(e.label||'')+'" style="'+style+'" edge="1" source="'+
      e.from+'" target="'+e.to+'" parent="1"><mxGeometry relative="1" as="geometry"><Array as="points">'+
      e.points.slice(1,-1).map(v=>'<mxPoint x="'+p(v[0])+'" y="'+p(v[1])+'"/>').join('')+
      '</Array></mxGeometry></mxCell>');
  }
  all.push('</root></mxGraphModel></diagram>');
  return all.join('\n');
}
async function main(){
  fs.mkdirSync(path.join(ROOT,'images'),{recursive:true});
  const manifest={date:spec.date,version:spec.version,retained_formats:['svg'],figures:[]};
  const pages=[];
  for(const [i,g]of spec.figures.entries()){
    validateLayout(g);
    for(const n of g.nodes){
      const metrics=await Promise.all(n.text.map(t=>sharp({text:{text:pango(t),font:'Songti SC 10.5',dpi:72,rgba:true}}).metadata()));
      const widths=metrics.map(x=>x.width);
      const limit=n.shape==='decision'?n.w*0.60:n.shape==='terminal'?n.w*0.85:n.w-12;
      if(Math.max(...widths)>limit)throw new Error(g.id+' text too wide in '+n.id+': '+widths+' > '+limit);
      n._text_widths=widths;
    }
    for(const e of g.edges)if(e.label){
      const m=await sharp({text:{text:pango(e.label),font:'Songti SC 10.5',dpi:72,rgba:true}}).metadata();
      e._label_width=m.width;
    }
    const svg=drawSvg(g);
    fs.writeFileSync(path.join(ROOT,'images',g.id+'.svg'),svg);
    pages.push(drawioPage(g,i+1));
    manifest.figures.push({id:g.id,title:g.title,
      svg_sha256:crypto.createHash('sha256').update(svg).digest('hex')});
    console.log(g.id+' SVG');
  }
  fs.writeFileSync(path.join(ROOT,'论文流程图.drawio'),'<mxfile host="app.diagrams.net">'+pages.join('\n')+'</mxfile>');
  fs.writeFileSync(path.join(ROOT,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
  // 图集正文独立维护，不再生成PNG、预览、HTML或逐图draw.io副本。
}
main().catch(err=>{console.error(err);process.exitCode=1;});
