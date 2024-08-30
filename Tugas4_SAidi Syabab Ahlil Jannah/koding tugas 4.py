import pandas as pd
import numpy as np

# Impor data
data = pd.DataFrame({
    'date': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'penetrasi': [52.5, 57.6, 62.2, 66.3, 69.9, 72.9, 75.3],
    'pengguna': [139000000, 154100000, 168300000, 181500000, 193200000, 203500000, 212200000]
})

# Pra-pemrosesan data
data['date'] = pd.to_datetime(data['date'], format='%Y')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Memisahkan data menjadi fitur dan target
X = data['date'].values
y = data['penetrasi'].values

# Membuat model LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# Membagi data menjadi data pelatihan dan pengujian
train_X, test_X = X[:-1], X[-1:]
train_y, test_y = y[:-1], y[-1:]

# Reshape data untuk sesuai dengan format LSTM
train_X = train_X.reshape((train_X.shape[0], 1, 1))

# Melatih model
model.fit(train_X, train_y, epochs=100, verbose=0)
from pykalman import UnscentedKalmanFilter

# Fungsi untuk memperkirakan ketidakpastian
def estimate_uncertainty(observation_noise, initial_state_mean, initial_state_covariance):
    kf = UnscentedKalmanFilter(
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        observation_covariance=observation_noise
    )
    (filtered_state_means, _) = kf.filter(y)
    return filtered_state_means

# Memperkirakan ketidakpastian
observation_noise = 0.01  # Anda dapat menyesuaikan ini
initial_state_mean = train_y[-1]
initial_state_covariance = 1.0

predicted_growth = estimate_uncertainty(observation_noise, initial_state_mean, initial_state_covariance)
import matplotlib.pyplot as plt

# Membuat grafik
plt.figure(figsize=(10, 6))
plt.plot(X, y, marker='o', label='Data Pengguna E-Commerce', color='blue')
plt.plot(X[-1:], predicted_growth, marker='x', label='Prediksi Pertumbuhan', color='red')
plt.fill_between(X[-1:], predicted_growth - 2 * observation_noise, predicted_growth + 2 * observation_noise, alpha=0.2, color='red', label='Ketidakpastian (95%)')
plt.xlabel('Tahun')
plt.ylabel('Penetrasi Pengguna E-Commerce')
plt.title('Prediksi Pertumbuhan Pengguna E-Commerce dengan Ketidakpastian')
plt.legend()
plt.show()
