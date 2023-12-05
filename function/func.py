import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie
import numpy as np
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

def tim_nguong(df):
      df['UP'] = df['UP'] /1000000
      df['DOWN'] = df['DOWN'] /1000000
      percentile = 76

      # Tính toán ngưỡng cho cả hai cột
      threshold_up = np.percentile(df['UP'], percentile)
      threshold_down = np.percentile(df['DOWN'], percentile)

      print("Ngưỡng cho cột UP:", threshold_up)
      print("Ngưỡng cho cột DOWN:", threshold_down)
      return threshold_up, threshold_down