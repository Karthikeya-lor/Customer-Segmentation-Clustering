// Portable evaluation of fitted sklearn parameters. No fitting or synthetic predictions.
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sum=a=>a.reduce((s,v)=>s+v,0);
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
const softmax=a=>{const z=Math.max(...a),v=a.map(x=>Math.exp(x-z)),s=sum(v);return v.map(x=>x/s)};
const argmax=a=>a.reduce((best,v,i)=>v>a[best]?i:best,0);
export function transform(op,x){
 switch(op.kind){
 case 'sequence': return op.steps.reduce((v,s)=>transform(s,v),x);
 case 'columns': return op.parts.flatMap(p=>transform(p.op,p.fields.map(f=>x[f])));
 case 'impute': return x.map((v,i)=>v==null||Number.isNaN(v)?op.fill[i]:v);
 case 'clip': return x.map((v,i)=>Math.max(op.low[i],Math.min(op.high[i],v)));
 case 'scale': return x.map((v,i)=>(v-op.mean[i])/op.scale[i]);
 case 'polynomial': return op.powers.map(p=>p.reduce((v,e,i)=>v*Math.pow(x[i],e),1));
 case 'onehot': return op.categories.flatMap((cats,i)=>cats.flatMap((c,j)=>op.drop&&op.drop[i]===j?[]:[x[i]===c?1:0]));
 default: throw Error('Unsupported transform '+op.kind);
 }
}
function leaf(t,x){let i=0;while(t.left[i]!==-1)i=Math.fround(x[t.feature[i]])<=t.threshold[i]?t.left[i]:t.right[i];return t.value[i]}
function svm(m,x){return m.intercept+m.support.reduce((s,v,i)=>s+m.dual[i]*(m.kernel==='linear'?dot(v,x):Math.exp(-m.gamma*sum(v.map((q,j)=>(q-x[j])**2)))),0)}
function svmProbability(decision,A,B){
 const r01=Math.min(1-1e-7,Math.max(1e-7,sigmoid(-(-decision*A+B)))),r10=1-r01;
 const Q=[[r10*r10,-r10*r01],[-r10*r01,r01*r01]],p=[.5,.5];
 for(let iteration=0;iteration<100;iteration++){
  let Qp=Q.map(row=>dot(row,p)),pQp=dot(p,Qp);
  if(Math.max(...Qp.map(v=>Math.abs(v-pQp)))<.0025)break;
  for(let t=0;t<2;t++){
   const diff=(-Qp[t]+pQp)/Q[t][t];p[t]+=diff;
   pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/((1+diff)**2);
   Qp=Qp.map((v,j)=>(v+diff*Q[t][j])/(1+diff));
   for(let j=0;j<2;j++)p[j]/=1+diff;
  }
 }
 return p;
}
function average(vectors){return vectors[0].map((_,i)=>sum(vectors.map(v=>v[i]))/vectors.length)}
export function evaluate(m,x){
 switch(m.kind){
 case 'linear': return {value:dot(m.coef,x)+m.intercept};
 case 'target_scaled': return {value:evaluate(m.model,x).value*m.scale+m.mean};
 case 'logistic': {const p=sigmoid(dot(m.coef,x)+m.intercept);return {probabilities:[1-p,p]}}
 case 'svr': return {value:svm(m,x)};
 case 'svc': {const d=svm(m,x);return {classIndex:d>=0?1:0,probabilities:svmProbability(d,m.A,m.B)}}
 case 'tree': return m.classification?{probabilities:leaf(m,x)}:{value:leaf(m,x)[0]};
 case 'forest': return m.classification?{probabilities:average(m.trees.map(t=>leaf(t,x)))}:{value:sum(m.trees.map(t=>leaf(t,x)[0]))/m.trees.length};
 case 'bagging': return {probabilities:average(m.trees.map((t,i)=>leaf(t,m.features[i].map(f=>x[f]))))};
 case 'boosting': {const v=m.initial+m.rate*sum(m.trees.map(t=>leaf(t,x)[0]));return m.classification?{probabilities:[1-sigmoid(v),sigmoid(v)]}:{value:v}}
 case 'adaboost': {const vote=sum(m.trees.map((t,i)=>(argmax(leaf(t,x))===1?1:-1)*m.weights[i]))/sum(m.weights),p=sigmoid(2*vote);return {probabilities:[1-p,p]}}
 case 'gaussian': {const log=m.mean.map((mu,k)=>Math.log(m.prior[k])-.5*sum(mu.map((v,i)=>Math.log(2*Math.PI*m.variance[k][i])+(x[i]-v)**2/m.variance[k][i])));return {probabilities:softmax(log)}}
 case 'mlp': {let v=x;for(let l=0;l<m.weights.length;l++){const w=m.weights[l];v=m.bias[l].map((b,j)=>b+sum(v.map((q,i)=>q*w[i][j])));v=v.map(z=>l===m.weights.length-1?sigmoid(z):m.activation==='tanh'?Math.tanh(z):Math.max(0,z))}return {probabilities:[1-v[0],v[0]]}}
 case 'knn': {
  const nearest=m.X.map((v,i)=>({i,d:sum(v.map((q,j)=>Math.abs(q-x[j])**m.p))**(1/m.p)})).sort((a,b)=>a.d-b.d||a.i-b.i).slice(0,m.neighbors);
  const zero=nearest.some(v=>v.d===0),weights=nearest.map(v=>m.weights==='distance'?(zero?(v.d===0?1:0):1/v.d):1),total=sum(weights);
  if(m.classification){const p=[0,0];nearest.forEach((v,j)=>p[m.y[v.i]]+=weights[j]/total);return {probabilities:p}}
  return {value:sum(nearest.map((v,j)=>m.y[v.i]*weights[j]))/total};
 }
 case 'kmeans': {const distances=m.centers.map(c=>Math.sqrt(sum(c.map((v,i)=>(v-x[i])**2))));return {value:distances.indexOf(Math.min(...distances)),distances}}
 default: throw Error('Unsupported estimator '+m.kind);
 }
}
export function predict(payload,record){
 let input={...record};
 const observed=v=>typeof v==='number'&&Number.isFinite(v);
 // Preserve missing derived values so the fitted imputer handles them as sklearn does.
 if(payload.track==='regression')input.volume=[input.x,input.y,input.z].every(observed)?input.x*input.y*input.z:null;
 if(payload.track==='classification')input.Compactness=[input.Area,input.Perimeter].every(observed)?4*Math.PI*input.Area/(input.Perimeter**2+1e-8):null;
 let x=payload.transforms[0].kind==='columns'?input:payload.fields.map(f=>input[f]);
 for(const op of payload.transforms)x=transform(op,x);
 const result=evaluate(payload.estimator,x);
 if(result.probabilities){result.classIndex??=argmax(result.probabilities);result.value=payload.classes[result.classIndex]}
 return result;
}
