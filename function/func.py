import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie

def load_css(file_name):
      with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)   

def load_lottiefile(filepath: str):
      with open(filepath, "r") as f:
            return json.load(f)
      
def load_lottieurl(url: str):
      r = requests.get(url)
      if r.status_code != 200:
            return None
      
      return r.json()