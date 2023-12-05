import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from drawchart import draw
def data_processing(df):
      df_makh = df['MAKHACHHANG']
      df.drop('MAKHACHHANG', axis = 'columns', inplace=True)
      df.drop('PHUONGXA', axis = 'columns', inplace=True)

      #lable encoder
      label_encoder = LabelEncoder()
      for column in df.select_dtypes(include='object').columns:
            df[column] = label_encoder.fit_transform(df[column])

      # sacle
      cols_to_scale = ['GIADICHVU']
      scaler = MinMaxScaler()
      df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

      # xu ly mat can bang
      sm = SMOTE()
      dataset_dummy = pd.get_dummies(df, drop_first=True)
      X = dataset_dummy.drop(["THANHLY"],axis=1)
      y = dataset_dummy['THANHLY']
      X_res, y_res = sm.fit_resample(X, y)
      X = X_res
      y = y_res

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
      result_df = pd.DataFrame({'MAKHACHHANG': df_makh, 'THANHLY': y}).reset_index(drop=True)
      result_df = result_df.loc[X_test.index]

    # Get MAKHACHHANG for the test set
      makh_test = result_df['MAKHACHHANG']

      return X_train, X_test, y_train, y_test, makh_test
      

def train_model(model_name, X_train, y_train, X_test, y_test, makh_test):
      if model_name == "Decision Tree":
            model = DecisionTreeClassifier()
      elif model_name == "KNN":
            model = KNeighborsClassifier()
      elif model_name == "Random Forest":
            model = RandomForestClassifier()

      # Huấn luyện mô hình
      model.fit(X_train, y_train)

      # Dự đoán và đánh giá mô hình trên tập kiểm tra
      y_pred = model.predict(X_test)

      st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
      st.subheader("Đánh giá kết quả:")
      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred)
      # Create a DataFrame to display the metrics
      metrics_data = {
            'Metric': ['F1', 'Precision', 'Recall'],
            'Score': [f"{round(f1, 2)*100}%", f"{round(precision, 2)*100}%", f"{round(recall,2)*100}%"]
      }

      metrics_df = pd.DataFrame(metrics_data)
      st.table(metrics_df)

      st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
      confusion_mat = confusion_matrix(y_test, y_pred)
      confusion_df = pd.DataFrame(confusion_mat, columns=["Thanh lý", "Không thanh lý"], index=["Thanh lý", "Không thanh lý"])
      
      st.subheader("Kết quả dự đoán:")
      st.dataframe(confusion_df)
      accuracy = accuracy_score(y_test, y_pred)
     

      st.markdown('<hr style="border:1px solid #F63366;">', unsafe_allow_html=True)
      
      #'Mã khách hàng': test_set_id,
      # Kết quả dự đoán của X_test
      y_test_labels = y_test.replace({0: "Yes", 1: "No"})
      y_pred_labels = pd.Series(y_pred).replace({0: "Yes", 1: "No"})
      result_df = pd.DataFrame({'Mã khách hàng': makh_test, 'Nhãn thực tế': y_test_labels.values, 'Nhãn dự đoán': y_pred_labels.values})
      st.subheader("File kết quả dự đoán:")
      result_df = result_df.dropna().reset_index(drop=True)
      result_df['Mã khách hàng'] = result_df['Mã khách hàng'].apply(lambda x: '{:,.0f}'.format(x))
      st.write(result_df)
      draw.download_csv_button(result_df, '⬇️ Tải xuống tại đây ', 'Ket_qua_du_doan')
      # st.write(len(result_df))
      # return result_df