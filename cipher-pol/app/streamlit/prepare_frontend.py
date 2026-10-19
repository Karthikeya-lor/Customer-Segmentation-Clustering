"""Package the original website unchanged, with only a Streamlit lifecycle bridge."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'app/site/dist'
TARGET = ROOT / 'app/streamlit/frontend'
BRIDGE_TAG = '<script src="streamlit-bridge.js"></script>'


def prepare():
    TARGET.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in sorted(SOURCE.rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(SOURCE)
        target = TARGET / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    original = (SOURCE / 'index.html').read_text(encoding='utf-8')
    (TARGET / 'index.html').write_text(original.replace('</body>', BRIDGE_TAG + '</body>'), encoding='utf-8')
    shutil.copyfile(ROOT / 'app/streamlit/streamlit-bridge.js', TARGET / 'streamlit-bridge.js')
    report = {'status': 'PASS', 'source': 'app/site/dist', 'identical_assets': len(hashes) - 1,
              'index_change': 'Only the Streamlit lifecycle script tag is appended.', 'source_sha256': hashes}
    for relative, expected in hashes.items():
        if relative != 'index.html':
            assert hashlib.sha256((TARGET / relative).read_bytes()).hexdigest() == expected, relative
    assert (TARGET / 'index.html').read_text(encoding='utf-8').replace(BRIDGE_TAG, '') == original
    (ROOT / 'app/streamlit/frontend-parity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    report = prepare()
    print(f"PASS: {report['identical_assets']} assets are byte-identical to the original website; original index preserved with lifecycle bridge.")
