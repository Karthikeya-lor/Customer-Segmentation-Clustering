# Cipher Pol — original interface on Streamlit

Live app: https://cipher-pol-learning-lab.streamlit.app/

Streamlit serves the complete original Cipher Pol website as a local custom component. The original typography, split particle hero, diamond/rice/customer graphics, track cards, four-stage workflow and all five lab sections are retained. No dependency on the previous Sites host is required. The custom component lifecycle bridge only enables hosting, viewport sizing, normal scrolling and shareable section anchors.

All 98 non-index frontend assets are byte-identical to `app/site/dist`, including the original CSS, JavaScript, 21 portable fitted models, result catalogues, all 60 figures and executed notebook downloads. The HTML adds only the lifecycle bridge script. Model inputs are evaluated locally in the browser using the audited fitted parameters; they are not uploaded to the Python server. The sklearn joblib artifacts and `app/inference.py` remain the numerical reference.

## Run

Use Python 3.14. In the curated GitHub `cipher-pol` folder:

```text
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

In the original workspace, install `app/streamlit/requirements.txt`, run `python app/streamlit/prepare_frontend.py`, then start the root entrypoint. Rebuild the frontend after changing `app/site/dist`; do not edit generated frontend files directly.

## Deployment

Repository: `Karthikeya-lor/Customer-Segmentation-Clustering`. Branch: `main`. Main file: `cipher-pol/streamlit_app.py`. Cloud Python: 3.14. Sharing remains **only specific people can view this app**. Public source publication was separately approved by the owner.

The bundle contains all website assets and model parameters. Existing original repository files are preserved. Hosting instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Verification

`frontend-parity.json` records SHA-256 identities. `verification.json` records the Streamlit host test. `exact-browser-checks.json` records original-versus-restored checks across 15 lab sections, 19 comparison views / 698 metric cells, 22 detail views, all 21 supported prediction paths, all plot galleries and four workflow stages, plus desktop/mobile layout checks. The complete scientific model audit is `app/deep-verification.json`; changing the host does not retrain models or change their results. Earlier native UI experiments are archived locally and are not the deployed interface.

Currency and rice variety names remain unspecified because verified mappings were not supplied. Agglomerative clustering has no native new-record prediction. Held-out scores describe the evaluated samples, not guaranteed future accuracy.

## Container alternative

From the curated `cipher-pol` folder:

```text
docker build -t cipher-pol .
docker run --rm -p 8501:8501 cipher-pol
```
