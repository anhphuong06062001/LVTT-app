from sklearn.metrics import confusion_matrix
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
from ml_processing import ml
############LOAD MODEL###############
model_up = load_model('model\lstm_model-UP-complete.h5')
scaler_up = joblib.load('model\scaler.joblib')

model_down = load_model('model\lstm_model-DOWN.h5')
scaler_down = joblib.load('model\scaler_down.joblib')
def create_time_series_data(data, column_name, id_col,date_col, date_churn, label_col, n_steps, n_forecast):
      X = []
      Y = []
      customer_ids = []
      ngay_kh = []
      ngay_thanhly = []
      nhan = []

      data[column_name] = data[column_name] / 1000000 
      data[column_name] = data[column_name].astype(int)
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
                  ngay_tl = customer_data[date_churn].iloc[i + n_steps:i + n_steps + n_forecast].to_numpy()
                  nhan_tl = customer_data[lable_col].iloc[i + n_steps:i + n_steps + n_forecast].to_numpy()
                  X.append(X_data)
                  Y.append(Y_data)
                  customer_ids.append(id)
                  ngay_kh.append(ngay)
                  ngay_thanhly.append(ngay_tl)
                  nhan.append(nhan_tl)
     
      X = np.array(X)
      Y = np.array(Y)
      customer_ids = np.array(customer_ids)
      ngay_kh = np.array(ngay_kh)
      ngay_thanhly = np.array(ngay_thanhly)
      nhan = np.array(nhan)
      return X, Y, customer_ids, ngay_kh, ngay_thanhly, nhan

def predict_churn_up(x_test, y_test, id_test, ngay_test, ngay_thanhly, nhan):
      predictions = model_up.predict(x_test)

      predictions_inverse = scaler_up.inverse_transform(predictions.reshape(-1, 1))
      y_test_inverse = scaler_up.inverse_transform(y_test.reshape(-1, 1))
      id_test = id_test.reshape(-1, 1)
      ngay_test = ngay_test.reshape(-1, 1)
      ngay_test = pd.to_datetime(ngay_test[:,0])
      ngay_thanhly_test = ngay_thanhly.reshape(-1,1)
      ngay_thanhly_test = pd.to_datetime(ngay_thanhly_test[:,0])
      nhan_up_test = nhan.reshape(-1,1)
      st.subheader(f'Kết quả dự đoán:')
      # result = pd.DataFrame(data={'KHACHHANG_ID': id_test[:,0],'NGAY': ngay_test,'Actuals': y_test_inverse[:,0], 'Predictions':predictions_inverse[:,0]})
      result = pd.DataFrame(data={'KHACHHANG_ID': id_test[:,0],'NGAY': ngay_test,'Actuals': y_test_inverse[:,0], 'Predictions':predictions_inverse[:,0], 'Ngay_ThanhLy': ngay_thanhly_test,'ThanhLy_Thucte':nhan_up_test[:,0]} )
      
      result['Difference'] = abs(result['Predictions'] - result['Actuals'])

      min_indices = result.groupby(['KHACHHANG_ID', 'NGAY'])['Difference'].idxmin()

      selected_rows = result.loc[min_indices]

      selected_rows = selected_rows.drop(columns=['Difference'])


      return selected_rows

def predict_churn_down(x_test, y_test, id_test, ngay_test, ngay_thanhly, nhan):
      predictions = model_down.predict(x_test)

      predictions_inverse = scaler_up.inverse_transform(predictions.reshape(-1, 1))
      y_test_inverse = scaler_up.inverse_transform(y_test.reshape(-1, 1))
      id_test = id_test.reshape(-1, 1)
      ngay_test = ngay_test.reshape(-1, 1)
      ngay_test = pd.to_datetime(ngay_test[:,0])
      ngay_thanhly_test = ngay_thanhly.reshape(-1,1)
      ngay_thanhly_test = pd.to_datetime(ngay_thanhly_test[:,0])
      nhan_up_test = nhan.reshape(-1,1)
      st.subheader(f'Kết quả dự đoán lưu lượng:')
      # result = pd.DataFrame(data={'KHACHHANG_ID': id_test[:,0],'NGAY': ngay_test,'Actuals': y_test_inverse[:,0], 'Predictions':predictions_inverse[:,0]})
      result = pd.DataFrame(data={'KHACHHANG_ID': id_test[:,0],'NGAY': ngay_test,'Actuals': y_test_inverse[:,0], 'Predictions':predictions_inverse[:,0], 'Ngay_ThanhLy': ngay_thanhly_test,'ThanhLy_Thucte':nhan_up_test[:,0]} )
      
      result['Difference'] = abs(result['Predictions'] - result['Actuals'])

      min_indices = result.groupby(['KHACHHANG_ID', 'NGAY'])['Difference'].idxmin()

      selected_rows = result.loc[min_indices]

      selected_rows = selected_rows.drop(columns=['Difference'])


      return selected_rows
def handle_after_predict(df, data, nguong):
      st.subheader(f'Kết quả dự đoán nhãn:')
      result_df = pd.DataFrame(columns=['KHACHHANG_ID', 'Trung binh'])
      unique_customers = df['KHACHHANG_ID'].unique()

      for customer_id in unique_customers:
            customer_data = df[df['KHACHHANG_ID'] == customer_id]
            max_date = customer_data['NGAY'].max()

            start_date = max_date - pd.DateOffset(days=7)

            filtered_data = customer_data[(customer_data['NGAY'] >= start_date) & (customer_data['NGAY'] <= max_date)]

            average_value = filtered_data['Predictions'].mean()

            result_df = result_df.append({'KHACHHANG_ID': customer_id, 'Trung binh': average_value}, ignore_index=True)

      result_df['THANHLY_DUDOAN'] = result_df['Trung binh'].apply(lambda x: 0 if x > nguong else 1)

      new_data_test = data.loc[:, ['KHACHHANG_ID', 'THANHLY']]
      grouped_data_test = new_data_test.groupby('KHACHHANG_ID').last()

      merged_test = grouped_data_test.merge(result_df, on='KHACHHANG_ID')
      merged_test =  merged_test.drop(['Trung binh'], axis=1)
      merged_test.replace({0: 'No', 1: 'Yes'}, inplace=True)
      return merged_test
############DỰNG LAYOUT, PHÂN TRANG, THAO TÁC###############
st.set_page_config(layout="wide")

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state.df = None

if 'df_ml' not in st.session_state:
    st.session_state.df_ml = None

if 'id_col' not in st.session_state:
    st.session_state.id_col = None

if 'custumer' not in st.session_state:
    st.session_state.custumer = None

if 'lable' not in st.session_state:
    st.session_state.lable = None

if 'date' not in st.session_state:
    st.session_state.date = None

if 'chart' not in st.session_state:
    st.session_state.chart = None

with st.sidebar:
      selected = option_menu(
            menu_title = "Main Menu",
            options = ['Home', 'Import Data', 'Draw Chart', 'Prediction LSTM','Orther model','Feedback'],
            icons=["house", "cloud-upload", "bar-chart-fill", "list-task","motherboard-fill", "people-fill"],
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
      with st.form(key='import_file', clear_on_submit=False):
            if uploaded_file is not None:
                  st.session_state.df = pd.read_csv(uploaded_file)  
                  
                  if st.session_state.df is not None and not st.session_state.df.empty:
                        columns = st.session_state.df.columns.tolist()
                        id_col = st.selectbox("Select data for id custumer", columns)
                        unique_customers = st.session_state.df[id_col].unique()  
                        lable_col = st.selectbox("Select data for lable", columns)
                        date_col = st.selectbox("Select data for date", columns)
                        if st.form_submit_button("Draw"):
                              st.session_state.chart = draw.show_data_info(st.session_state.df, unique_customers ,lable_col, id_col, date_col)
                  #     st.session_state.show = kq
      
            else:
                  if st.session_state.df is not None and not st.session_state.df.empty:
                        columns = st.session_state.df.columns.tolist()
                        st.session_state.id_col = st.selectbox("Select data for id custumer", columns)
                        st.session_state.custumer = st.session_state.df[st.session_state.id_col].unique()  
                        st.session_state.lable = st.selectbox("Select data for lable", columns)
                        st.session_state.date = st.selectbox("Select data for date", columns)
                        
                        if st.form_submit_button("Draw"):
                              st.session_state.chart = draw.show_data_info(st.session_state.df, st.session_state.custumer ,st.session_state.lable, st.session_state.id_col, st.session_state.date)
      # st.write(st.session_state.chart)
                  
elif selected == 'Draw Chart':
      # session_state.page = 'Draw Chart'
      st.title('Draw Chart')
      with st.form(key="draw_chart"):
            if st.session_state.df is not None and not st.session_state.df.empty:
                  columns = st.session_state.df.columns.tolist()

                  x_axis = st.selectbox("Select data for X axis", columns)
                  y_axis = st.selectbox("Select data for Y axis", columns)

                  customer_id_column = st.selectbox("Select column for Customer ID", columns)

                  unique_customers = st.session_state.df[customer_id_column].unique()  
                  num_customers = st.slider("Select number of customers", min_value=1, max_value=len(unique_customers), value=10)

                  if st.form_submit_button("Draw Chart"):
                        selected_customers = unique_customers[:num_customers]

                        
                        filtered_data = st.session_state.df[st.session_state.df[customer_id_column].isin(selected_customers)]

                        final_data = pd.DataFrame()
                        for customer_id in selected_customers:
                              final_data = pd.concat([final_data, filtered_data[filtered_data[customer_id_column] == customer_id]])
                        st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
                        draw.plot_customer_data(final_data, x_axis, y_axis, customer_id_column)
                        st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
                        draw.trend_month(final_data, x_axis, y_axis)
                        st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
                        draw.trend_per_month(final_data, x_axis, y_axis)
            

elif selected == 'Prediction LSTM':
      # session_state.page = 'Prediction'
      st.title('Prediction')
      with st.form(key="prediction"):
            if st.session_state.df is not None and not st.session_state.df.empty:
                  columns = st.session_state.df.columns.tolist()

                  data_column = st.selectbox("Select data for predict", columns)
                  customer_id_column = st.selectbox("Select column for Customer ID", columns)
                  date_column = st.selectbox("Select data for time serise", columns)
                  date_churn = st.selectbox("Select data for date churn", columns)
                  lable_col = st.selectbox("Select data for label", columns)
                  selected_model = st.selectbox("Select Machine Learning Model", ["LSTM for Upload", "LSTM for Download"])
                  
                  if st.form_submit_button('Predict'):
                        st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
                        X_test, y_test, id_test, ngay_test, ngay_thanhly, nhan = create_time_series_data(st.session_state.df, data_column, customer_id_column, date_column, date_churn, lable_col, 23, 7)
                        X_test_up = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))
                        
                        nguong_up, nguong_down = func.tim_nguong(st.session_state.df)

                        if selected_model == "LSTM for Upload":
                              kq = predict_churn_up(X_test_up, y_test, id_test, ngay_test, ngay_thanhly, nhan)
                              st.write(kq)
                              draw.download_csv_button(kq, '⬇️ Tải xuống tại đây', 'kq_dudoan_luuluong')
                             
                              final_res = handle_after_predict(kq, st.session_state.df, nguong_up)
                              st.write(final_res)
                              st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
                              st.subheader("Kết quả dự đoán:")
                              st.write(confusion_matrix(final_res['THANHLY'], final_res['THANHLY_DUDOAN']))
                              draw.download_csv_button(final_res, '⬇️ Tải xuống tại đây', 'kq_dudoan_nhan')
                        elif selected_model == "LSTM for Download":
                              kq = predict_churn_down(X_test_up, y_test, id_test, ngay_test, ngay_thanhly, nhan)
                              st.write(kq)
                              draw.download_csv_button(kq, '⬇️ Tải xuống tại đây ', 'kq_dudoan_luuluong')
                              final_res = handle_after_predict(kq, st.session_state.df, nguong_down)
                              st.write(final_res)
                              st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
                              st.subheader("Kết quả dự đoán:")
                              st.write(confusion_matrix(final_res['THANHLY'], final_res['THANHLY_DUDOAN']))
                              draw.download_csv_button(final_res, '⬇ Tải xuống tại đây', 'kq_dudoan_nhan')
elif selected == 'Orther model':
      st.header("Model Machine learning")
      uploaded_file = st.file_uploader("Upload file CSV or Excel", type=['csv', 'xlsx'])

      with st.form(key="model"):
            if uploaded_file is not None:
                  st.session_state.df_ml = pd.read_csv(uploaded_file) 
                  st.write('Preview of the imported data:')
                  st.write(st.session_state.df_ml.head()) 
                  X_train, X_test, y_train, y_test, makh_test = ml.data_processing(st.session_state.df_ml)
                  selected_model = st.selectbox("Select Model", ["Decision Tree", "KNN", "Random Forest"])
      
                  if st.form_submit_button('Train'):
                        if selected_model == "Decision Tree":
                              ml.train_model(selected_model, X_train, y_train, X_test, y_test, makh_test)
                        elif selected_model == "KNN":
                              ml.train_model(selected_model, X_train, y_train, X_test, y_test, makh_test)
                        elif selected_model == "Random Forest":
                              ml.train_model(selected_model, X_train, y_train, X_test, y_test, makh_test)
                              
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
