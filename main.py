import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from streamlit_option_menu import option_menu
from function import func
from streamlit_lottie import st_lottie
from drawchart import draw
############LOAD MODEL###############
model_up = load_model('model\lstm_model-v2.h5')
scaler_up = joblib.load('model\scaler.joblib')

model_down = load_model('model\lstm_model-DOWN.h5')
scaler_down = joblib.load('model\scaler_down.joblib')
def create_time_series_data(data, column_name, id_col,date_col,n_steps, n_forecast):
      X = []
      Y = []
      customer_ids = []
      ngay_kh = []

      data[column_name] = data[column_name] / 1000000 
      data[column_name] = scaler_up.fit_transform(data[column_name].values.reshape(-1, 1))

      for customer_id, group in data.groupby(id_col):
            customer_data = group.sort_values(by=date_col)
            data_length = len(customer_data)

            for i in range(data_length - n_steps - n_forecast + 1):
            # Xử lý dữ liệu và chia thành X và Y
                  X_data = customer_data[column_name].iloc[i:i + n_steps].to_numpy()
                  Y_data = customer_data[column_name].iloc[i + n_steps:i + n_steps + n_forecast].to_numpy()
                  id = customer_data[id_col].iloc[i + n_steps:i + n_steps + n_forecast].to_numpy()
                  ngay = customer_data[date_col].iloc[i + n_steps:i + n_steps + n_forecast].to_numpy()
           
                  X.append(X_data)
                  Y.append(Y_data)
                  customer_ids.append(id)
                  ngay_kh.append(ngay)
     
      X = np.array(X)
      Y = np.array(Y)
      customer_ids = np.array(customer_ids)
      ngay_kh = np.array(ngay_kh)
      return X, Y, customer_ids, ngay_kh

def predict_churn_up(x_test, y_test, id_test, ngay_test):
      predictions = model_up.predict(x_test)

      predictions_inverse = scaler_up.inverse_transform(predictions.reshape(-1, 1))
      y_test_inverse = scaler_up.inverse_transform(y_test.reshape(-1, 1))
      id_test = id_test.reshape(-1, 1)
      ngay_test = ngay_test.reshape(-1, 1)
      ngay_test = pd.to_datetime(ngay_test[:,0])
      st.subheader(f'Kết quả dự đoán:')
      result = pd.DataFrame(data={'KHACHHANG_ID': id_test[:,0],'NGAY': ngay_test,'Actuals': y_test_inverse[:,0], 'Predictions':predictions_inverse[:,0]})

      return result

def predict_churn_down(x_test, y_test, id_test, ngay_test):
      predictions = model_down.predict(x_test)

      predictions_inverse = scaler_down.inverse_transform(predictions.reshape(-1, 1))
      y_test_inverse = scaler_down.inverse_transform(y_test.reshape(-1, 1))
      id_test = id_test.reshape(-1, 1)
      ngay_test = ngay_test.reshape(-1, 1)
      ngay_test = pd.to_datetime(ngay_test[:,0])
      st.subheader(f'Kết quả dự đoán:')
      result = pd.DataFrame(data={'KHACHHANG_ID': id_test[:,0],'NGAY': ngay_test,'Actuals': y_test_inverse[:,0], 'Predictions':predictions_inverse[:,0]})

      return result
############HÀM VẼ###############


############DỰNG LAYOUT, PHÂN TRANG, THAO TÁC###############
st.set_page_config(layout="wide")



if 'df' not in st.session_state:
    st.session_state.df = None


with st.sidebar:
      selected = option_menu(
            menu_title = "Main Menu",
            options = ['Home', 'Import Data', 'Draw Chart', 'Prediction', 'Feedback'],
            icons=["house", "cloud-upload", "envelope", "list-task", "people-fill"],
            menu_icon="cast",
      )

if selected == 'Home':
      # session_state.page = 'Home'
      st.title('Welcome to Customer Churn Prediction App')
      st.write('This app helps you analyze and predict customer churn.')
      lottie_hello = func.load_lottiefile("jsonlottie/hello.json")
      st_lottie(
           lottie_hello,
           speed=1, 
           reverse=False,
           loop=True, 
           quality="medium",
           height=500,
           width=1000, 
           key=None  
      )
      

elif selected == 'Import Data':   
      # session_state.page = 'Import Data'
      st.title('Import Data')
      uploaded_file = st.file_uploader("Upload file CSV or Excel", type=['csv', 'xlsx'])
      #st.session_state.show = ''      
      if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)  
            
            if st.session_state.df is not None and not st.session_state.df.empty:
                  columns = st.session_state.df.columns.tolist()
                  id_col = st.selectbox("Select data for id custumer", columns)
                  unique_customers = st.session_state.df[id_col].unique()  
                  lable_col = st.selectbox("Select data for lable", columns)
                  date_col = st.selectbox("Select data for date", columns)
                  if st.button("Draw"):
                      draw.show_data_info(st.session_state.df, unique_customers ,lable_col, id_col, date_col)
                  #     st.session_state.show = kq
      
      
      # st.write(st.session_state.show)
                  
elif selected == 'Draw Chart':
      # session_state.page = 'Draw Chart'
      st.title('Draw Chart')
      if st.session_state.df is not None and not st.session_state.df.empty:
            columns = st.session_state.df.columns.tolist()

            x_axis = st.selectbox("Select data for X axis", columns)
            y_axis = st.selectbox("Select data for Y axis", columns)

            customer_id_column = st.selectbox("Select column for Customer ID", columns)

            unique_customers = st.session_state.df[customer_id_column].unique()  
            num_customers = st.slider("Select number of customers", min_value=1, max_value=len(unique_customers), value=10)

            if st.button("Draw Chart"):
                  selected_customers = unique_customers[:num_customers]

                  
                  filtered_data = st.session_state.df[st.session_state.df[customer_id_column].isin(selected_customers)]

                  final_data = pd.DataFrame()
                  for customer_id in selected_customers:
                        final_data = pd.concat([final_data, filtered_data[filtered_data[customer_id_column] == customer_id]])
                  st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
                  draw.plot_customer_data(final_data, x_axis, y_axis, customer_id_column)
                  st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
                  draw.trend_month(final_data, x_axis, y_axis)
            

elif selected == 'Prediction':
      # session_state.page = 'Prediction'
      st.title('Prediction')
      if st.session_state.df is not None and not st.session_state.df.empty:
            columns = st.session_state.df.columns.tolist()

            data_column = st.selectbox("Select data for predict", columns)
            customer_id_column = st.selectbox("Select column for Customer ID", columns)
            date_column = st.selectbox("Select data for time serise", columns)
            selected_model = st.selectbox("Select Machine Learning Model", ["LSTM for Upload", "LSTM for Download"])

            if st.button('Predict'):
                  st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
                  X_test, y_test, id_test, ngay_test = create_time_series_data(st.session_state.df, data_column, customer_id_column, date_column, 23, 7)
                  X_test_up = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))
                  
                  if selected_model == "LSTM for Upload":
                        kq = predict_churn_up(X_test_up, y_test, id_test, ngay_test)
                        st.write(kq)
                  elif selected_model == "LSTM for Download":
                        kq = predict_churn_down(X_test_up, y_test, id_test, ngay_test)
                        st.write(kq)

elif selected == 'Feedback':
      st.header(":mailbox: Please give me your comments to help me improve the application!!!")

      contact_form = """
            <form action="https://formsubmit.co/anhphuong06062001@gmail.com" method="POST">
                  <input type="hidden" name="_captcha" value="false">
                  <input type="text" name="name" placeholder="Your name" required>
                  <input type="email" name="email" placeholder="Your email" required>
                  <textarea name="message" placeholder="Detail your feedback"></textarea>
                  <button type="submit">Send</button>
            </form>
      """

      st.markdown(contact_form, unsafe_allow_html=True)

      func.load_css("css/style.css")

      lottie_coding = func.load_lottiefile("jsonlottie/coding.json")
      st_lottie(
           lottie_coding,
           speed=1, 
           reverse=False,
           loop=True, 
           quality="medium",
           height=200,
           width=1800, 
           key=None  
      )
