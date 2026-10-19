"""Cipher Pol: native Streamlit UI with inference through the verified sklearn artifacts."""
from pathlib import Path
import json
import html
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from app.inference import predict_regression, predict_classification, assign_cluster

DIST = ROOT / 'app/site/dist'
st.set_page_config(page_title='Cipher Pol — Learning Lab', page_icon='◈', layout='wide')
st.html((ROOT / 'app/streamlit/style.css').read_text(encoding='utf-8'))


@st.cache_data
def data_for(track):
    return json.loads((DIST / f'data/{track}.json').read_text(encoding='utf-8'))


@st.cache_data
def catalogue():
    return json.loads((DIST / 'data/catalog.json').read_text(encoding='utf-8'))


def fmt(value, metric):
    if pd.isna(value): return '—'
    unit = catalogue()['metric_definitions'].get(metric, {}).get('unit')
    return f'{value:.2%}' if unit == 'fraction' else f'{value:,.2f}' if metric in ['RMSE','MAE'] else f'{value:,.4f}'


def plot(fig):
    fig.update_layout(template='plotly_white', font_family='Arial', margin=dict(l=25,r=25,t=45,b=35),
        colorway=['#151715','#84968b','#9e8272','#626f87','#9c915c'], paper_bgcolor='white', plot_bgcolor='white')
    st.plotly_chart(fig, width='stretch')


def metrics(row, names):
    for col, name in zip(st.columns(len(names)), names): col.metric(name.replace('F1_weighted','Weighted F1'), fmt(row[name],name))


def downloads(d):
    st.divider()
    for col, (label, file, mime) in zip(st.columns(4), [
        ('Comparison CSV',d['track']+'_comparison.csv','text/csv'),
        ('Executed notebook',d['track']+'.ipynb','application/x-ipynb+json'),
        ('Verification report','verification.json','application/json'), ('Project notes','README.md','text/markdown')]):
        path = DIST / 'downloads' / file
        payload=path.read_bytes()
        if file=='verification.json' and (ROOT/'app/streamlit/verification.json').exists():
            evidence=json.loads(payload);evidence['streamlit_migration']=json.loads((ROOT/'app/streamlit/verification.json').read_text())
            payload=json.dumps(evidence,indent=2).encode('utf-8')
        col.download_button(label,payload,file_name=file,mime=mime,key='download_'+file)


def overview(d):
    selected = d.get('selection',{}).get('selected_model',d.get('selection',{}).get('model','K-Means'))
    m = next(m for m in d['models'] if m['name']==selected)
    st.subheader(d['subject']+'. A question worth exploring.')
    st.caption('K-Means at a glance' if d['track']=='clustering' else 'CV-selected model: '+selected)
    metrics(m['metrics'],d['metrics'])
    a,b,c = st.columns(3)
    with a:
        st.markdown('**01 / THE DATA**')
        st.write(f"{d['dataset']['rows']:,} source rows · {d['dataset']['cleaned_rows']:,} cleaned rows")
        st.write('Discover customer groups without campaign outcome labels.' if d['track']=='clustering' else
            'Predict diamond price from dimensions and quality.' if d['track']=='regression' else 'Distinguish rice classes using grain geometry.')
    with b:
        st.markdown('**02 / THE METHOD**')
        st.write('Same imputation, IQR clipping, scaling and encoding for both methods.' if d['track']=='clustering' else
            'Seed 42 · shared 80:20 split · preprocessing refitted inside every training CV fold.')
    with c:
        st.markdown('**03 / THE EVIDENCE**')
        st.write(f"{len(d['models'])} algorithms · {len(d['plots'])} verified figures")
        st.write('Compare several metrics, inspect errors, and try an actual fitted model.')
    with st.expander('Evaluation and preprocessing'):
        if d['track']=='clustering':
            st.write('Clustering metrics use the full transformed cohort. PCA/t-SNE are visual projections. Five-fold subset refits measure stability, not held-out predictive accuracy. Campaign outcomes are excluded from fitting.')
            st.write('Historical age/tenure reference: '+d['reference_date']+'. Agglomerative has no native prediction for a new record.')
        else:
            st.write('Hyperparameters and the deployment model are selected using five-fold training CV. Test metrics use reserved rows. Reused tuning folds are not an independent nested-CV estimate.')
            st.write('Numeric features use fitted median imputation, IQR clipping and scaling. '+('Quality categories are one-hot encoded. Volume = x × y × z. SVR also scales its target inside training folds.' if d['track']=='regression' else 'Compactness = 4π × Area / (Perimeter² + 10⁻⁸). Precision, recall and F1 are weighted by true class support.'))
        st.dataframe(pd.DataFrame(catalogue()['metric_definitions']).T, width='stretch')
        st.code('Dataset SHA-256: '+d['dataset']['sha256'])
    st.markdown('#### A look at the source data')
    st.dataframe(pd.DataFrame(d['dataset']['preview']), hide_index=True, width='stretch')


def comparison(d):
    st.subheader('Different approaches. One fair comparison.')
    clust = d['track']=='clustering'
    metric = st.selectbox('Compare by',d['metrics'],key='metric_'+d['track'])
    definition = catalogue()['metric_definitions'][metric]
    lower = definition['direction']=='lower'
    st.caption(('Lower is better ↓' if lower else 'Higher is better ↑')+' · '+definition['definition'])
    stage = 'Tuned' if clust else st.radio('Evaluation stage',['Tuned','Baseline'],horizontal=True)
    chosen = st.multiselect('Models',[m['name'] for m in d['models']],default=[m['name'] for m in d['models']],key='models_'+d['track'])
    rows=[]
    for m in d['models']:
        row={'Model':m['name'],**{k:m['metrics'][k] for k in d['metrics']}}
        if stage=='Baseline': row.update({v['Metric']:v['Baseline'] for v in m['tuning']})
        rows.append(row)
    full=pd.DataFrame(rows).set_index('Model')
    table=full.loc[chosen].sort_values(metric,ascending=lower,kind='stable')
    if len(table):
        plot(px.bar(table.reset_index(),x=metric,y='Model',orientation='h',title=stage+' / '+metric).update_yaxes(autorange='reversed'))
        st.dataframe(table.style.format({m:lambda v,m=m:fmt(v,m) for m in d['metrics']}),width='stretch')
    else: st.info('Choose at least one model to compare.')
    st.markdown('#### A closer comparison')
    a,b=st.columns(2)
    first=a.selectbox('First model',full.index,key='pair_a_'+d['track'])
    second=b.selectbox('Second model',full.index,index=1,key='pair_b_'+d['track'])
    diff=full.loc[first,metric]-full.loc[second,metric]
    st.write(f'{first}: **{fmt(full.loc[first,metric],metric)}** · {second}: **{fmt(full.loc[second,metric],metric)}**')
    if abs(diff)<1e-12: st.caption('The models have the same score for this metric.')
    else:
        better=first if (diff<0 if lower else diff>0) else second
        unit=' percentage points' if definition['unit']=='fraction' else ''
        st.caption(f"Absolute difference: {abs(diff)*(100 if unit else 1):.4f}{unit}. {better} has the better {metric} score.")
    if clust:
        st.markdown('#### What changes with linkage?')
        st.dataframe(pd.DataFrame(d['linkages']),hide_index=True,width='stretch')
        st.caption(d['linkage_selection_rule'])
    else: st.caption('Tuning uses training CV. A tuned model can improve one held-out metric while worsening another; all changes remain visible.')


def detail(d):
    name=st.selectbox('Model detail',[m['name'] for m in d['models']],key='detail_'+d['track'])
    m=next(m for m in d['models'] if m['name']==name)
    metrics(m['metrics'],d['metrics'])
    if d['track']=='clustering':
        st.markdown('#### Customer profiles in original feature units')
        st.dataframe(pd.DataFrame(m['profiles']),hide_index=True,width='stretch')
        st.caption('Age and tenure reference '+d['reference_date']+'. Cluster numbers identify fitted groups, not invented personas.')
        plot(px.bar(x=list(m['cluster_sizes']),y=list(m['cluster_sizes'].values()),labels={'x':'Cluster','y':'Customers'},title='Cluster sizes'))
        st.markdown('#### Five-fold subset refits')
        st.dataframe(pd.DataFrame(m['resampling']),hide_index=True,width='stretch')
        st.caption('Preprocessing and clustering are refitted on each 80% subset. Adjusted Rand compares subset labels with full-fit labels. These are stability checks, not supervised test accuracy.')
        if name=='K-Means': plot(px.line(pd.DataFrame(d['elbow']),x='k',y='Inertia',markers=True,title='K-Means elbow / selected k='+str(m['metrics']['Clusters'])))
        else:
            st.dataframe(pd.DataFrame(d['linkages']),hide_index=True,width='stretch')
            st.caption('Selected linkage: '+m['metrics']['Linkage']+'. '+d['linkage_selection_rule'])
        return
    st.markdown('#### Selected hyperparameters')
    st.json(m['parameters'])
    st.markdown('#### Before and after tuning')
    st.dataframe(pd.DataFrame(m['tuning']),hide_index=True,width='stretch')
    p=pd.DataFrame(m['predictions'])
    a,b=st.columns(2)
    if d['track']=='regression':
        with a:
            fig=px.scatter(p,x='actual',y='predicted',labels={'actual':'Actual price','predicted':'Predicted price'},title='Held-out predicted vs actual')
            low,high=p[['actual','predicted']].min().min(),p[['actual','predicted']].max().max()
            fig.add_shape(type='line',x0=low,y0=low,x1=high,y1=high,line_dash='dash');plot(fig)
        with b:
            p['residual']=p.actual-p.predicted
            plot(px.scatter(p,x='predicted',y='residual',title='Held-out residuals').add_hline(y=0,line_dash='dash'))
    else:
        with a: plot(px.imshow(confusion_matrix(p.actual,p.predicted,labels=d['class_labels']),x=d['class_labels'],y=d['class_labels'],text_auto=True,color_continuous_scale='Greys',labels={'x':'Predicted class','y':'Actual class'},title='Held-out confusion matrix'))
        with b:
            fpr,tpr,_=roc_curve(p.actual,p.probability_1)
            plot(px.line(x=fpr,y=tpr,labels={'x':'False positive rate','y':'True positive rate'},title='Held-out ROC / AUC '+fmt(m['metrics']['ROC_AUC'],'ROC_AUC')).add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line_dash='dash'))
    if 'coefficients' in m:
        st.markdown('#### Fitted coefficients')
        st.dataframe(pd.DataFrame(m['coefficients']),hide_index=True,width='stretch')
        st.caption('Conditional on other transformed inputs. Numeric inputs are scaled; category coefficients refer to encoded levels. Odds ratios apply only to logistic regression.')
        if 'zero_coefficient_count' in m: st.caption(f"Near-zero coefficients: {m['zero_coefficient_count']}")
    if 'feature_importance' in m:
        frame=pd.DataFrame(m['feature_importance']).sort_values('Importance',ascending=False)
        plot(px.bar(frame.head(15),x='Importance',y='Feature',orientation='h',title='Fitted feature importance').update_yaxes(autorange='reversed'))
        st.caption('Split usefulness in this fitted model, not a causal effect. Correlated inputs can share importance.')
    if 'training_loss_curve' in m: plot(px.line(x=np.arange(1,len(m['training_loss_curve'])+1),y=m['training_loss_curve'],labels={'x':'Iteration','y':'Training loss'},title='MLP training loss'))
    best=max(m['search'],key=lambda r:r['mean_test_score'])
    st.markdown('#### Five-fold training validation')
    st.dataframe(pd.DataFrame({'Fold':range(1,6),'Score':[best[f'split{i}_test_score'] for i in range(5)]}),hide_index=True)
    st.caption('Training CV '+('R²' if d['track']=='regression' else 'weighted F1')+'. These folds were reused for tuning; this is not nested CV.')
    with st.expander(f"Full search results / {len(m['search'])} configurations"):
        st.dataframe(pd.DataFrame(m['search'])[['params','mean_test_score','std_test_score','rank_test_score']],hide_index=True,width='stretch')


def gallery(d):
    st.subheader('A different view of the evidence.')
    st.caption(f"{len(d['plots'])} actual figures from the executed notebooks. PCA/t-SNE are projections; cluster metrics use the full transformed feature space.")
    titles={p['name'].replace('_',' ').title():p for p in d['plots']}
    title=st.selectbox('Figure',list(titles),key='plot_'+d['track'])
    item=titles[title];path=DIST/item['url']
    st.image(str(path),caption=title,width='stretch')
    st.download_button('Download this figure',path.read_bytes(),file_name=path.name,mime='image/png')
    with st.expander('All figure thumbnails'):
        cols=st.columns(3)
        for i,p in enumerate(d['plots']): cols[i%3].image(str(DIST/p['url']),caption=p['name'].replace('_',' '),width='stretch')


def predictions(d):
    clust=d['track']=='clustering'
    st.subheader('Your inputs. A real model result.')
    st.caption('Predictions run on the Streamlit Python server using the actual fitted sklearn pipeline. Inputs are sent to this server; this app does not write them to files.')
    choices=['K-Means'] if clust else [m['name'] for m in d['models']]
    selected=d.get('selection',{}).get('selected_model',d.get('selection',{}).get('model','K-Means'))
    name=st.selectbox('Predict with',choices,index=choices.index(selected),key='predict_model_'+d['track'])
    if clust: st.info('Historical age/tenure reference: '+d['reference_date']+'. Agglomerative has no native new-record prediction; its cohort results remain available in Model detail.')
    if st.button('Load dataset example',key='example_'+d['track']):
        st.session_state['example_mode_'+d['track']]=True
        st.session_state['input_generation_'+d['track']]=st.session_state.get('input_generation_'+d['track'],0)+1
    with st.form('predict_'+d['track']):
        cols=st.columns(3);record={}
        counts={'Kidhome','Teenhome','Recency','NumDealsPurchases','NumWebVisitsMonth','Customer_Tenure_Days','Total_Purchases'}
        for i,field in enumerate(d['input_fields']):
            f=field['name'];key='input_'+d['track']+'_'+f+'_'+str(st.session_state.get('input_generation_'+d['track'],0));label=f.replace('_',' ').replace('AspectRation','Aspect ratio')
            default=d['example_input'][f] if st.session_state.get('example_mode_'+d['track']) else field['default']
            with cols[i%3]:
                if field['type']=='category': record[f]=st.selectbox(label,field['choices'],index=field['choices'].index(default),key=key)
                elif clust and f in counts: record[f]=st.number_input(label,min_value=0,value=int(default),step=1,key=key)
                else: record[f]=st.number_input(label,min_value=0.0,value=float(default),step=0.01,format='%.6f',key=key)
        submitted=st.form_submit_button('Find segment' if clust else 'Run prediction',type='primary')
    if submitted:
        try:
            if d['track']=='regression':
                result=predict_regression(record,name);st.metric('Estimated diamond price',f"{result['predicted_price']:,.2f}");st.caption(result['price_unit'])
            elif d['track']=='classification':
                result=predict_classification(record,name);st.metric('Predicted rice class',str(result['predicted_class']))
                st.dataframe(pd.DataFrame([{'Class':k,'Probability':v} for k,v in result['probabilities'].items()]).style.format({'Probability':'{:.2%}'}),hide_index=True)
                st.caption('Labels remain 0 and 1; a verified variety-name mapping was not supplied. Probabilities are model estimates, not guarantees.')
            else:
                result=assign_cluster(record);st.metric('Customer segment',str(result['cluster']))
                st.dataframe(pd.DataFrame({'Cluster':range(len(result['centroid_distances'])),'Centroid distance':result['centroid_distances']}),hide_index=True)
                st.caption(result['note'])
            st.success('Evaluated with '+name+' and its saved fitted preprocessing.')
        except ValueError as error: st.error(str(error))
    st.caption('Change inputs or the model, then submit again to see a new result. Results clear on the next interaction to avoid stale output.')


st.html('<div class="cipher-nav"><b>⠿ CIPHER POL</b><span>TRACKS &nbsp;&nbsp; MODELS &nbsp;&nbsp; EVIDENCE</span><small>23CSE301 / LEARNING LAB</small></div>')
left,right=st.columns([1.1,1])
with left:
    st.html('<div class="cipher-hero"><div class="cipher-kicker">APPLIED MACHINE LEARNING / VOL. 01</div><h1>Understand<br>intelligence.<br><span>One model<br>at a time.</span></h1><p>Three real datasets. Twenty-two trained models.<br>Explore the evidence, compare the trade-offs,<br>and turn your inputs into predictions.</p><a href="#cipher-lab">ENTER THE LAB ↗</a></div>')
with right:
    rng=np.random.default_rng(42);t=rng.uniform(0,2*np.pi,5000);u=rng.uniform(0,2*np.pi,5000)
    r=.42;cx=(2+np.cos(3*t))*np.cos(2*t);cy=(2+np.cos(3*t))*np.sin(2*t);cz=np.sin(3*t)
    fig=go.Figure(go.Scatter3d(x=cx+r*np.cos(u)*np.cos(2*t),y=cy+r*np.cos(u)*np.sin(2*t),z=cz+r*np.sin(u),mode='markers',marker=dict(size=1.2,color='#161816',opacity=.65)))
    fig.update_layout(height=480,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='white',scene=dict(xaxis_visible=False,yaxis_visible=False,zaxis_visible=False,aspectmode='data',camera=dict(eye=dict(x=1.4,y=1.4,z=.8))),showlegend=False)
    st.plotly_chart(fig,width='stretch',config={'displayModeBar':False},key='hero_geometry')
    st.html('<div class="cipher-terminal">$ cipher-pol / learning pipeline<br>✓ 10 regressors evaluated<br>✓ 10 classifiers compared<br>✓ 02 clustering methods fitted<br>✓ actual Python model inference</div>')
st.html('<div class="cipher-strip">◇ DIAMONDS &nbsp; / &nbsp; ⋮ RICE GRAINS &nbsp; / &nbsp; ◎ CUSTOMERS &nbsp; / &nbsp; CROSS-VALIDATION &nbsp; / &nbsp; PYTHON INFERENCE</div><div id="cipher-lab"></div><div class="cipher-kicker">THE EVIDENCE / AT YOUR FINGERTIPS</div>')
track_label=st.radio('Problem track',['Diamonds · Regression','Rice · Classification','Marketing · Clustering'],horizontal=True,key='track')
track={'Diamonds · Regression':'regression','Rice · Classification':'classification','Marketing · Clustering':'clustering'}[track_label]
d=data_for(track);ds=d['dataset']
st.caption(f"{d['subject'].upper()} / RAW {ds['rows']:,} / CLEANED {ds['cleaned_rows']:,} / "+(f"COHORT {ds['fitting_rows']:,}" if track=='clustering' else f"TRAIN {ds['training_rows']:,} · TEST {ds['test_rows']:,}")+f" / MODELS {len(d['models'])}")
section=st.radio('Lab section',['Overview','Compare models','Model detail','Plot explorer','Try a prediction'],horizontal=True,key='section')
{'Overview':overview,'Compare models':comparison,'Model detail':detail,'Plot explorer':gallery,'Try a prediction':predictions}[section](d)
downloads(d)
st.html('<div class="cipher-footer">CIPHER POL / 23CSE301 &nbsp; · &nbsp; DATA → PREPARE → FIT → EVALUATE → EXPLORE</div>')
