import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import uuid

# Initialize session state for model-related variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'testPredict' not in st.session_state:
    st.session_state.testPredict = None
if 'testY_actual' not in st.session_state:
    st.session_state.testY_actual = None

# Fungsi untuk membuat dataset dengan look_back
def create_dataset(X, Y, look_back=1):
    Xs, Ys = [], [] 
    for i in range(len(X) - look_back - 1):
        Xs.append(X[i:(i + look_back)])
        Ys.append(Y[i + look_back])
    return np.array(Xs), np.array(Ys)

# Model LSTM
def build_lstm_model(input_shape, units, dropout):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True, name=uuid.uuid4().hex))
    model.add(Dropout(dropout, name=uuid.uuid4().hex))
    model.add(LSTM(units, name=uuid.uuid4().hex))
    model.add(Dropout(dropout, name=uuid.uuid4().hex))
    model.add(Dense(1, name=uuid.uuid4().hex))
    model.compile(loss='mse', optimizer='adam')
    return model

# Fungsi untuk membuat plot prediksi harga Bitcoin
def plot_predictions(actual_prices, predicted_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='red')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# Load data
df = pd.read_excel("DATA FIKS.xlsx")
dataY = df['Price'].values.reshape(-1, 1)
dataX = df.drop(['Date', 'Price'], axis=1).values  # drop columns you don't want
feature_names = df.drop(['Date', 'Price'], axis=1).columns.tolist()

# Standarisasi data
scalerX, scalerY = StandardScaler(), StandardScaler()
dataX = scalerX.fit_transform(dataX)
dataY = scalerY.fit_transform(dataY)

# Buat model
look_back = 1
train_size = int(len(dataX) * 0.8)
trainX, testX = dataX[:train_size], dataX[train_size:]
trainY, testY = dataY[:train_size], dataY[train_size:]

trainX, trainY = create_dataset(trainX, trainY, look_back)
testX, testY = create_dataset(testX, testY, look_back)

# Bentuk ulang input menjadi 3D
trainX = trainX.reshape(trainX.shape[0], look_back, trainX.shape[2])
testX = testX.reshape(testX.shape[0], look_back, testX.shape[2])

def build_and_train_model():
    st.session_state.model = None
    st.session_state.testY_actual = None
    st.session_state.testPredict = None

    st.session_state.harga_btc = 0.00
    st.session_state.volume = 0.00
    st.session_state.nasdaq = 0.00
    st.session_state.harga_emas = 0.00
    st.session_state.sp_500 = 0.00

    # Get selected features from session state
    selected_features = st.session_state.selected_features

    # Update dataX sesuai variabel yang dipilih
    dataX_selected = df[selected_features].values
    dataX_selected_scaled = scalerX.fit_transform(dataX_selected)

    # Bagi data ke dalam train/test
    trainX_selected, testX_selected = dataX_selected_scaled[:train_size], dataX_selected_scaled[train_size:]
    trainX_selected, trainY_selected = create_dataset(trainX_selected, dataY[:train_size], look_back)
    testX_selected, testY_selected = create_dataset(testX_selected, dataY[train_size:], look_back)

    trainX_selected = trainX_selected.reshape(trainX_selected.shape[0], look_back, trainX_selected.shape[2])
    testX_selected = testX_selected.reshape(testX_selected.shape[0], look_back, testX_selected.shape[2])

    # Buat model
    st.session_state.model = build_lstm_model((look_back, trainX_selected.shape[2]), units=100, dropout=0.3)
    st.session_state.model.fit(trainX_selected, trainY_selected, epochs=50, batch_size=16, verbose=0, shuffle=False)

    # Prediksi menggunakan data testing
    testPredict_scaled = st.session_state.model.predict(testX_selected)
    st.session_state.testPredict = scalerY.inverse_transform(testPredict_scaled)
    st.session_state.testY_actual = scalerY.inverse_transform(testY_selected)


def prediksi_dashboard_menu():
    # Dashboard menggunakan Streamlit
    st.title("Prediksi Harga Bitcoin")

    # Multiselect widget to choose input variables
    selected_features = st.multiselect(
        'Pilih variabel input', 
        feature_names, 
        key='selected_features', 
        on_change=build_and_train_model  # Automatically calls the function when the selection changes
    )

    # Jika tidak ada variabel yang dipilih, berikan pesan
    if len(selected_features) == 0:
        st.write("Silakan pilih setidaknya satu variabel untuk memulai prediksi.")
    else:
        if st.session_state.testY_actual is not None and st.session_state.testPredict is not None:
            
            # Hitung MAPE
            test_mape = mean_absolute_percentage_error(st.session_state.testY_actual, st.session_state.testPredict)
            st.write(f"Loss (MAPE) pada data testing: {test_mape:.2f}")

            # Plot hasil prediksi vs aktual
            st.subheader("Grafik Prediksi vs Data Aktual")
            fig, ax = plt.subplots()
            ax.plot(st.session_state.testY_actual, label="Data Aktual")
            ax.plot(st.session_state.testPredict, label="Prediksi")
            ax.legend()
            st.pyplot(fig)

            # SHAP interpretation berdasarkan data testing (not implemented yet)
            # st.subheader("Grafik SHAP untuk Interpretasi Model pada Data Testing")
            # fig, ax = plt.subplots()
            # st.pyplot(fig)

            # Prediksi 1 periode ke depan berdasarkan harga dan volume perdagangan
            st.subheader("Prediksi 1 Hari ke Depan")

            selected_indices = [False, False, False, False, False]

            # Input pengguna untuk prediksi 1 hari ke depan
            if 'Price_Lag1' in st.session_state.selected_features:
                harga_btc_t1 = st.number_input('Harga Bitcoin t-1', value=st.session_state.harga_btc, format="%.2f")
                selected_indices[3] = True
            else:
                harga_btc_t1 = 0

            if 'Vol' in st.session_state.selected_features:
                volume_btc = st.number_input('Volume Perdagangan Bitcoin', value=st.session_state.volume, format="%.2f")
                selected_indices[4] = True
            else:
                volume_btc = 0

            if 'Nasdaq' in st.session_state.selected_features:
                nasdaq = st.number_input('Valuasi Nasdaq', value=st.session_state.nasdaq, format="%.2f")
                selected_indices[1] = True
            else:
                nasdaq = 0

            if 'Harga_Emas' in st.session_state.selected_features:
                harga_emas = st.number_input('Harga Pasar Emas', value=st.session_state.harga_emas, format="%.2f")
                selected_indices[0] = True
            else:
                harga_emas = 0

            if 'S&P_500' in st.session_state.selected_features:
                sp_500 = st.number_input('S&P 500', value=st.session_state.sp_500, format="%.2f")
                selected_indices[2] = True
            else: 
                sp_500 = 0

            # Bentuk array input untuk prediksi 1 hari ke depan
            # Create the input array from user input values
            next_input = np.array([harga_emas, nasdaq, sp_500, harga_btc_t1, volume_btc])

            # Scale the input using the scaler
            next_input_scaled = scalerX.transform(next_input.reshape(1, -1)).reshape(1, 1, next_input.shape[-1])

            # Select only the features that are chosen by the user
            next_input_model = np.array([next_input_scaled[0, 0, i] for i in range(next_input_scaled.shape[2]) if selected_indices[i]]).reshape(1, 1, -1)

            # Prediksi 1 hari ke depan
            next_prediction_scaled = st.session_state.model.predict(next_input_model)
            next_prediction = scalerY.inverse_transform(next_prediction_scaled)

            st.write(f"Prediksi Harga Bitcoin 1 Hari ke Depan: {next_prediction[0][0]:.2f}")

def analysis_data_menu():
    st.title("Analisis Data Harga Bitcoin Menggunakan LSTM")
    
    # Variabel input yang diperlukan
    st.header('Pilih Variabel Input')
    input_vars = ['Gold Price', 'Nasdaq', 'S&P 500', 'Bitcoin Volume', 'Bitcoin Price']
    
    # Memilih variabel input
    selected_vars = st.multiselect('Pilih variabel input', input_vars)
    
    # Instruksi pengguna untuk mengunggah data
    st.write("Silakan unggah file CSV atau Excel berisi data dengan kolom yang sesuai.")
    
    # Input file CSV atau Excel
    uploaded_file = st.file_uploader("Upload file CSV atau Excel berisi data", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Membaca file yang diunggah pengguna
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Data yang diunggah:")
            st.write(df)
            
            # Memastikan kolom yang dipilih ada di data
            if all(var in df.columns for var in selected_vars):
                # Mengambil data sesuai variabel yang dipilih
                dataX = df[selected_vars[:-1]].values  # Semua kecuali kolom target
                dataY = df[selected_vars[-1]].values.reshape(-1, 1)  # Kolom target

                # Preprocessing: Standarisasi data
                scalerX = StandardScaler()
                dataX_scaled = scalerX.fit_transform(dataX)
                scalerY = StandardScaler()
                dataY_scaled = scalerY.fit_transform(dataY)

                # Membuat dataset untuk model
                look_back = 1
                dataX_prepared = create_dataset(dataX_scaled, look_back)
                dataY_prepared = dataY_scaled[look_back:]
                
                # Reshape input menjadi [samples, time steps, features]
                dataX_prepared = dataX_prepared.reshape(dataX_prepared.shape[0], look_back, len(selected_vars) - 1)
                
                # Split data ke train dan test
                train_size = int(len(dataX_prepared) * 0.8)
                trainX, testX = dataX_prepared[:train_size], dataX_prepared[train_size:]
                trainY, testY = dataY_prepared[:train_size], dataY_prepared[train_size:]
                
                # Membangun dan melatih model LSTM
                model = build_lstm_model((look_back, len(selected_vars) - 1), units=100, dropout=0.3)
                model.fit(trainX, trainY, epochs=50, batch_size=16, verbose=0, shuffle=False)
                
                # Prediksi menggunakan data testing
                predicted_scaled = model.predict(testX)
                predicted_prices = scalerY.inverse_transform(predicted_scaled)
                actual_prices = scalerY.inverse_transform(testY)
                
                # Tampilkan hasil prediksi
                st.subheader("Grafik Prediksi vs Data Aktual")
                plot_predictions(actual_prices, predicted_prices)
            
            else:
                st.error("Kolom yang diperlukan tidak ditemukan pada file yang diunggah.")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam membaca file: {e}")
    
    else:
        st.write("Silakan unggah file dengan kolom yang sesuai.")

# Streamlit Sidebar untuk Navigasi
st.sidebar.title("Drawer Menu")
option = st.sidebar.selectbox('Pilih opsi dari drawer', ['Home', 'Dashboard', 'Analisis Data', 'Profil'])

st.sidebar.write("Opsi yang dipilih:", option)

# Menampilkan konten sesuai pilihan user
if option == 'Home':
    st.title("Informasi tentang Aplikasi")
    st.write("Selamat datang di aplikasi prediksi harga Bitcoin!")

elif option == 'Dashboard':
    prediksi_dashboard_menu()

elif option == 'Analisis Data':
    analysis_data_menu()  # Panggil fungsi submenu analisis data

elif option == 'Profile':
    st.title("Profil Penulis")
    st.write("Ini adalah halaman tentang Penulis.")

# cara run nya   --streamlit run ./dashboard.py--
# semangat semhasnya hehehe
