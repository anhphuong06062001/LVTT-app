import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from streamlit_option_menu import option_menu


############LOAD MODEL###############
model = load_model('model\lstm_model-v1.h5')
scaler = joblib.load('model\scaler.joblib')

def create_time_series_data(data, column_name, id_col,date_col,n_steps, n_forecast):
      X = []
      Y = []
      customer_ids = []
      ngay_kh = []

      data[column_name] = data[column_name] / 1000000 
      data[column_name] = scaler.fit_transform(data[column_name].values.reshape(-1, 1))

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

def predict_churn(x_test, y_test, id_test, ngay_test):
      predictions = model.predict(x_test)

      predictions_inverse = scaler.inverse_transform(predictions.reshape(-1, 1))
      y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
      id_test = id_test_up.reshape(-1, 1)
      ngay_test = ngay_test_up.reshape(-1, 1)
      ngay_test = pd.to_datetime(ngay_test[:,0])
      st.subheader(f'Kết quả dự đoán:')
      result = pd.DataFrame(data={'KHACHHANG_ID': id_test[:,0],'NGAY': ngay_test,'Actuals': y_test_inverse[:,0], 'Predictions':predictions_inverse[:,0]})

      return result
############HÀM VẼ###############
def plot_customer_data(data, x_column, y_column, customer_id_column):
      st.subheader(f'Biểu đồ biến động lưu lượng {y_column} theo {x_column}')
      unique_customers = data[customer_id_column].unique()
      data[x_column] = pd.to_datetime(data[x_column])
      fig, ax = plt.subplots(figsize=(12, 6))
      for customer_id in unique_customers:
            customer_data = data[data[customer_id_column] == customer_id]
            ax.plot(customer_data[x_column], customer_data[y_column], label=f'Customer {customer_id}')

      ax.set_xlabel(x_column)
      ax.set_ylabel(y_column)
      ax.xaxis.set_major_locator(plt.MaxNLocator(10))
      fig.autofmt_xdate() 

      st.pyplot(fig)

def trend_month(data, x_column, columns):
      st.subheader(f'Biểu đồ biến động lưu lượng {columns} trong từng tháng')
      monthly_average_up = data.resample('M', on=x_column)[columns].mean()
      
      fig, ax = plt.subplots(figsize=(12, 6))
      ax.plot(monthly_average_up.index, monthly_average_up.values, marker='o', linestyle='-')
      ax.set_xlabel("Tháng")
      ax.set_ylabel("Trung bình Lưu lượng UP")
      ax.grid(True)
      plt.xticks(rotation=45)

      st.pyplot(fig)

def show_data_info(data, unique_customers, label_column, id_col):
      st.subheader("Một số thông tin về dữ liệu")
      st.write(f"Số lượng khách hàng: ", len(unique_customers))

      if label_column:
            data_draw = data.groupby(id_col).last()
            st.write("Số lượng khách thanh lý và không thanh lý")
            label_counts = data_draw[label_column].value_counts()
            st.write(label_counts)

            fig, ax = plt.subplots(figsize=(8, 4))
            label_counts.plot(kind='bar', ax=ax)
            st.pyplot(fig)


############DỰNG LAYOUT, PHÂN TRANG, THAO TÁC###############
st.set_page_config(layout="wide")


if 'df' not in st.session_state:
    st.session_state.df = None



# page = st.sidebar.selectbox(
#     'Navigation',
#     ['Home', 'Import Data', 'Draw Chart', 'Prediction'],
# )

with st.sidebar:
      selected = option_menu(
            menu_title = "Main Menu",
            options = ['Home', 'Import Data', 'Draw Chart', 'Prediction'],
            icons=["house", "cloud-upload", "envelope", "list-task"],
            menu_icon="cast",
      )

if selected == 'Home':
      # session_state.page = 'Home'
      st.title('Welcome to Customer Churn Prediction App')
      st.write('This app helps you analyze and predict customer churn.')

elif selected == 'Import Data':   
      # session_state.page = 'Import Data'
      st.title('Import Data')
      uploaded_file = st.file_uploader("Upload file CSV or Excel", type=['csv', 'xlsx'])
            
      if uploaded_file is not None:
            
            st.session_state.df = pd.read_csv(uploaded_file)  
            st.write(st.session_state.df) 
            if st.session_state.df is not None and not st.session_state.df.empty:
                  columns = st.session_state.df.columns.tolist()
                  id_col = st.selectbox("Select data for id custumer", columns)
                  unique_customers = st.session_state.df[id_col].unique()  
                  lable_col = st.selectbox("Select data for lable", columns)
                  if st.button("Draw"):
                        show_data_info(st.session_state.df, unique_customers ,lable_col, id_col)
                  
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

                  plot_customer_data(final_data, x_axis, y_axis, customer_id_column)
                  trend_month(final_data, x_axis, y_axis)
            

elif selected == 'Prediction':
      # session_state.page = 'Prediction'
      st.title('Prediction')
      if st.session_state.df is not None and not st.session_state.df.empty:
            columns = st.session_state.df.columns.tolist()

            data_column = st.selectbox("Select data for predict", columns)
            customer_id_column = st.selectbox("Select column for Customer ID", columns)
            date_column = st.selectbox("Select data for time serise", columns)
            if st.button('Predict'):
                  X_UP_test, y_UP_test, id_test_up, ngay_test_up = create_time_series_data(st.session_state.df, data_column, customer_id_column, date_column, 23, 7)
                  X_test_up = np.reshape(X_UP_test, (X_UP_test.shape[0],X_UP_test.shape[1], 1))
                  kq = predict_churn(X_test_up, y_UP_test, id_test_up, ngay_test_up)
                  st.write(kq)
      
