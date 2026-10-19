const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
let seed=42;const rand=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296};
const gaussian=()=>Math.sqrt(-2*Math.log(Math.max(1e-9,rand())))*Math.cos(2*Math.PI*rand());
function points(kind,n){const result=[];for(let i=0;i<n;i++){
 const t=rand()*Math.PI*2,u=rand()*Math.PI*2;
 if(kind==='torus'){
  const r=2+Math.cos(3*t),tube=.29+rand()*.2;
  result.push([(r+tube*Math.cos(u))*Math.cos(2*t),(r+tube*Math.cos(u))*Math.sin(2*t),Math.sin(3*t)+tube*Math.sin(u)]);
 }else if(kind==='diamond'){
  const face=Math.floor(rand()*8),a=face*Math.PI/4,b=(face+1)*Math.PI/4,v=rand(),q=rand();
  const px=Math.cos(a)*(1-v)+Math.cos(b)*v,pz=Math.sin(a)*(1-v)+Math.sin(b)*v;
  if(i%3===0)result.push([px*(1-q),.05+q*1.7,pz*(1-q)]);
  else result.push([px*(1-.46*q),.05-q*.72,pz*(1-.46*q)]);
 }else if(kind==='rice'){
  const group=i%3,z=2*rand()-1,r=Math.sqrt(1-z*z),x=r*Math.cos(t)*.38,y=z*1.4;
  const angle=[-.4,.4,1.1][group];
  result.push([x*Math.cos(angle)-y*Math.sin(angle)+(group-1)*.9,x*Math.sin(angle)+y*Math.cos(angle)+(group===1?-.4:.2),r*Math.sin(t)*.28]);
 }else if(kind==='clusters'){
  const g=i%3;result.push([Math.cos(g*2.1)*1.6+gaussian()*.4,Math.sin(g*2.1)*1.6+gaussian()*.4,gaussian()*.45]);
 }else {const z=2*rand()-1,r=Math.sqrt(1-z*z);result.push([r*Math.cos(t)*1.4,z*1.4,r*Math.sin(t)*1.4])}
}return result}
export function startParticles(){
 document.querySelectorAll('canvas[data-particles]').forEach(canvas=>{
  const ctx=canvas.getContext('2d'),type=canvas.dataset.particles,hero=type==='hero',network=type==='network';
  const n=hero?11000:type==='sphere'?1400:1800;
  const modes=['torus','diamond','rice','clusters'],clouds=(hero?modes:[type==='network'?'sphere':type]).map(m=>points(m,n));
  let active=0,angle=hero?.6:.2,tilt=hero?.6:-.25,visible=true,w=0,h=0,last=0,drag=false,moved=false,startX=0,startY=0;
  let current=clouds[0].map(p=>[...p]);
  const resize=()=>{const box=canvas.getBoundingClientRect();w=box.width;h=box.height;const ratio=Math.min(devicePixelRatio||1,2);canvas.width=w*ratio;canvas.height=h*ratio;ctx.setTransform(ratio,0,0,ratio,0,0)};
  new ResizeObserver(resize).observe(canvas);new IntersectionObserver(entries=>visible=entries[0].isIntersecting).observe(canvas);
  if(hero){const button=canvas.parentElement;
   button.addEventListener('pointerdown',e=>{drag=true;moved=false;startX=e.clientX;startY=e.clientY;button.setPointerCapture(e.pointerId)});
   button.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-startX,dy=e.clientY-startY;if(Math.abs(dx)+Math.abs(dy)>3)moved=true;angle+=dx*.008;tilt+=dy*.005;startX=e.clientX;startY=e.clientY});
   button.addEventListener('pointerup',()=>drag=false);button.addEventListener('pointercancel',()=>drag=false);
   button.addEventListener('click',event=>{if(moved&&event.detail>0)return;active=(active+1)%4;document.querySelector('#figure-label').textContent=['FIG. 01 — TORUS KNOT (2,3)','FIG. 02 — DIAMOND / REGRESSION','FIG. 03 — RICE / CLASSIFICATION','FIG. 04 — CUSTOMERS / CLUSTERING'][active]});
  }
  function project(p,scale){const c=Math.cos(angle),s=Math.sin(angle),ct=Math.cos(tilt),st=Math.sin(tilt);const x=p[0]*c+p[2]*s,z=-p[0]*s+p[2]*c,y=p[1]*ct-z*st,zz=p[1]*st+z*ct;return [w/2+x*scale,h*.46+y*scale,zz]}
  function draw(now){requestAnimationFrame(draw);if(!visible||!w||now-last<33)return;last=now;ctx.clearRect(0,0,w,h);if(!reduced&&!drag)angle+=hero?.003:.002;
   const scale=hero?Math.min(w/7.3,h/6.3):Math.min(w,h)/(type==='rice'?4.4:3.8);
   if(network){drawNetwork(ctx,w,h,angle);return}
   const target=clouds[active];
   for(let i=0;i<n;i++){
    for(let a=0;a<3;a++)current[i][a]+=(target[i][a]-current[i][a])*(reduced?1:.05);
    const [x,y,z]=project(current[i],scale),depth=(z+3.6)/7.2;
    ctx.fillStyle=`rgba(10,13,9,${Math.max(.14,Math.min(.9,.4+depth*.85))})`;
    const size=hero?.8+depth*1.1:.5+depth*.4;ctx.fillRect(x,y,size,size);
   }
  }
  requestAnimationFrame(draw);
 });
}
function drawNetwork(ctx,w,h,a){
 const cx=w/2,cy=h*.47,r=Math.min(w,h)*.34,nodes=Array.from({length:6},(_,i)=>[cx+Math.cos(i*Math.PI/3+a*.2)*r,cy+Math.sin(i*Math.PI/3+a*.2)*r*.8]);
 ctx.strokeStyle='#c1c4bb';ctx.lineWidth=.65;
 nodes.forEach(([x,y],i)=>{ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);ctx.stroke();ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(...nodes[(i+1)%6]);ctx.stroke();ctx.setLineDash([])});
 ctx.fillStyle='#fff';ctx.strokeStyle='#222';ctx.lineWidth=.7;
 nodes.forEach(([x,y],i)=>{ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#111';ctx.font='5px monospace';ctx.textAlign='center';ctx.fillText('A'+(i+1),x,y+2);ctx.fillStyle='#fff'});
 ctx.beginPath();ctx.arc(cx,cy,17,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#111';for(let x=-2;x<=2;x++)for(let y=-2;y<=2;y++)ctx.fillRect(cx+x*3,cy+y*3,1,1);
}
