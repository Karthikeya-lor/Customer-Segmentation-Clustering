"""Verify the exact website assets and Streamlit host registration.
Browser interaction checks are recorded separately in exact-browser-checks.json.
"""
from pathlib import Path
import hashlib
import json
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
report = json.loads((ROOT/'app/streamlit/frontend-parity.json').read_text())
frontend = ROOT/'app/streamlit/frontend'
for relative, expected in report['source_sha256'].items():
    content = (frontend/relative).read_bytes()
    if relative == 'index.html':
        original = (ROOT/'app/site/dist/index.html').read_text(encoding='utf-8')
        assert (frontend/relative).read_text(encoding='utf-8').replace('<script src="streamlit-bridge.js"></script>', '') == original
    else:
        assert hashlib.sha256(content).hexdigest() == expected, relative
at = AppTest.from_file(str(ROOT/'streamlit_app.py'), default_timeout=30).run()
assert not at.exception, [e.message for e in at.exception]
assert len(at.get('component_instance')) == 1
assert not at.error
result = {'status':'PASS', 'original_assets_byte_identical':report['identical_assets'],
          'index_only_adds_host_lifecycle_bridge':True, 'streamlit_host_registration':'PASS',
          'browser_checks':'app/streamlit/exact-browser-checks.json',
          'model_audit':'app/deep-verification.json',
          'portable_inference_audit':'app/site/model-parity-report.json'}
(ROOT/'app/streamlit/verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
