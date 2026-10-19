"""Exercise native Streamlit interactions and compare server predictions with verified expectations."""
from pathlib import Path
import json
import sys
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[2]
report={'status':'RUNNING','comparison_views':0,'model_detail_views':0,'prediction_paths':0,'track_sections':0}
at=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()

def good():
    assert not at.exception, '\n'.join(e.message for e in at.exception)

good()
expect=json.loads((ROOT/'app/streamlit/prediction-expectations.json').read_text())
for track,label in [('regression','Diamonds · Regression'),('classification','Rice · Classification'),('clustering','Marketing · Clustering')]:
    at.radio(key='track').set_value(label).run();good()
    d=json.loads((ROOT/f'app/site/dist/data/{track}.json').read_text())
    for section in ['Overview','Compare models','Model detail','Plot explorer','Try a prediction']:
        at.radio(key='section').set_value(section).run();good();report['track_sections']+=1
    at.radio(key='section').set_value('Compare models').run();good()
    for stage in ['Tuned'] if track=='clustering' else ['Tuned','Baseline']:
        if track!='clustering': at.radio[2].set_value(stage).run();good()
        for metric in d['metrics']:
            at.selectbox(key='metric_'+track).set_value(metric).run();good()
            table=at.dataframe[0].value
            for m in d['models']:
                for key in d['metrics']:
                    value=m['metrics'][key] if stage=='Tuned' else next(v['Baseline'] for v in m['tuning'] if v['Metric']==key)
                    np.testing.assert_allclose(table.loc[m['name'],key],value,atol=1e-12,rtol=0)
            lower=metric in ['RMSE','MAE','Davies-Bouldin Index']
            values=table[metric].to_numpy();assert np.all(np.diff(values)>=0) if lower else np.all(np.diff(values)<=0)
            report['comparison_views']+=1
    at.radio(key='section').set_value('Model detail').run();good()
    for m in d['models']:
        at.selectbox(key='detail_'+track).set_value(m['name']).run();good();report['model_detail_views']+=1
        if 'coefficients' in m:
            coefficient=next(df.value for df in at.dataframe if 'Coefficient' in df.value.columns)
            assert ('Odds_ratio' in coefficient)==(m['name']=='Logistic Regression')
    at.radio(key='section').set_value('Try a prediction').run();good()
    at.button(key='example_'+track).click().run();good()
    for item in expect[track]:
        at.selectbox(key='predict_model_'+track).set_value(item['name']).run();good()
        at.button(key='predict_submit_'+track).click().run();good()
        assert not at.error, [e.value for e in at.error]
        expected=f"{item['value']:,.2f}" if track=='regression' else str(int(item['value']))
        assert at.metric[0].value==expected,(track,item['name'],at.metric[0].value,expected)
        if track=='classification': np.testing.assert_allclose(at.dataframe[0].value.Probability,item['probabilities'],atol=1e-12,rtol=0)
        if track=='clustering': np.testing.assert_allclose(at.dataframe[0].value['Centroid distance'],item['distances'],atol=1e-12,rtol=0)
        report['prediction_paths']+=1
    # A widget interaction clears an earlier submitted result, so it cannot be mistaken for a new prediction.
    at.selectbox(key='predict_model_'+track).set_value(expect[track][0]['name']).run();good()
    assert not at.metric, 'Stale prediction remains visible'
    at.button(key='predict_submit_'+track).click().run();good()
    assert at.metric
    at.number_input[0].set_value(at.number_input[0].value+1).run();good()
    assert not at.metric, 'Prediction was not cleared after a numeric input edit'
    print('PASS',track,flush=True)
report['status']='PASS'
report['model_changes_clear_results']=True
report['numeric_input_edits_clear_results']=True
(ROOT/'app/streamlit/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
