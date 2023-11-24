import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

def show_data_info(data, unique_customers, label_column, id_col, date_col):
      st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
      st.subheader("Dữ liệu vừa import")
      st.write(data)
      st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
      st.subheader("Một số thông tin về dữ liệu")
      st.write(f" - Số lượng khách hàng: ", len(unique_customers))
      st.write(f" - Khoảng thời gian từ `{data[date_col].min()}` đến `{data[date_col].max()}`")

      if label_column:
            data_draw = data.groupby(id_col).last()
            st.write("Số lượng khách thanh lý và không thanh lý")
            label_counts = data_draw[label_column].value_counts()
            st.write(label_counts)
            st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
            st.subheader("Biểu đồ minh họa")
            fig, ax = plt.subplots(figsize=(8, 4))
            label_counts.plot(kind='bar', ax=ax)
            st.pyplot(fig)
