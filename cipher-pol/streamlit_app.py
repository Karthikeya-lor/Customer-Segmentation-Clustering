"""Serve the exact Cipher Pol website as a self-contained Streamlit component."""
from pathlib import Path
import streamlit as st
from streamlit.components.v1 import declare_component

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / 'app/streamlit/frontend'
st.set_page_config(page_title='Cipher Pol — Learning Lab', page_icon='◈', layout='wide')
st.html((ROOT / 'app/streamlit/host.css').read_text(encoding='utf-8'))

if not (FRONTEND / 'index.html').exists():
    st.error('The website bundle is missing. Run: python app/streamlit/prepare_frontend.py')
    st.stop()

# Streamlit serves every asset locally; this does not embed the previous hosted site.
website = declare_component('cipher_pol_website', path=str(FRONTEND))
website(key='cipher_pol_website', default=None)
