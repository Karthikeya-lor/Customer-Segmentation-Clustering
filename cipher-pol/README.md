# Cipher Pol — Streamlit Learning Lab

Native Streamlit app for diamond regression, rice classification and customer segmentation. It uses the verified sklearn joblib pipelines for predictions on the Python server. It includes all 22 model comparisons, baseline/tuned metrics, cross-validation, coefficients/importance, interactive diagnostics, all 60 notebook figures, model-input forms and downloads. The original project-wide audit is included; Streamlit migration checks are recorded separately.

## Run

Use Python 3.14, matching the verified training environment:

```text
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

In the original workspace, install from `app/streamlit/requirements.txt` instead. The application entrypoint is at the workspace root. The curated GitHub deployment is self-contained under `cipher-pol/`.

## Deploy on Streamlit Community Cloud

Repository: `Karthikeya-lor/Customer-Segmentation-Clustering`.
Branch: `main`.
Main file: `cipher-pol/streamlit_app.py`.
Python: **3.14** in Advanced settings. The requirements next to the entrypoint pin the versions used for inference. Set sharing to **private / only specific people** before handing off the app URL. The public repository does not provide private source-code storage; the owner explicitly approved publication of these deployment files.

Streamlit deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
Private sharing: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app

Inputs are sent to the Streamlit Python server. The app does not persist them to files. Currency and rice variety names remain unspecified because verified mappings were not supplied. Agglomerative has no native new-record prediction. Test-set metrics describe held-out samples, not guaranteed future accuracy.

## Container alternative

From the curated `cipher-pol` folder:

```text
docker build -t cipher-pol .
docker run --rm -p 8501:8501 cipher-pol
```

This starts Streamlit on port 8501. Configure access controls on the chosen hosting platform before external publication.
