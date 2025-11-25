# ⚡ Appliances Energy Prediction using Deep Learning

[](https://www.python.org/)
[](https://www.tensorflow.org/)
[](https://www.google.com/search?q=)

## 📖 Project Overview

This project focuses on forecasting the energy consumption of household appliances using historical sensor data from a low-energy building. The goal is to predict appliance energy usage (in Wh) based on environmental parameters such as temperature, humidity, and weather conditions.

The project compares traditional Machine Learning baselines (**Linear Regression**, **Random Forest**) against advanced Deep Learning architectures (**LSTM**, **GRU**, and a hybrid **CNN-LSTM**).

## 📊 Dataset

The dataset used is the **Appliances Energy Prediction Data Set**

  * **Time step:** 10 minutes.
  * **Duration:** 4.5 months.
  * **Target Variable:** `Appliances` (Energy usage in Wh).
  * **Features:** Indoor/Outdoor Temperature & Humidity, Windspeed, Visibility, Dewpoint.

## 📂 Project Structure

```text
Appliances-Energy-Prediction/
│
├── src/
│   │── energy_data.csv          # Raw dataset
│   └── energy_data_processed.py # Processed dataset
│
├── notebooks/
│   └── project.ipynb  # Jupyter Notebook with full analysis and training
│
├── models/
│   ├── scaler.pkl               # Saved MinMaxScaler for inference
│   └── cnn_lstm_model.h5        # Best performing Deep Learning model
│
├── Reports/                     
│   ├── Report.pdf               # Report of the entire project
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🛠️ Methodology

### 1\. Preprocessing

  * **Log Transformation:** Applied `np.log1p` to the target variable `Appliances` to handle significant right-skewness.
  * **Scaling:** Used `MinMaxScaler` to normalize all features to the range $[0, 1]$.
  * **Splitting:** Employed a time-series split (First 80% for Training, Last 20% for Testing) to preserve temporal order.

### 2\. Feature Engineering

To capture temporal dependencies, specific features were engineered:

  * **Temporal:** Hour of day, Day of week, Weekend flag.
  * **Lags:** Lag features at t-1, t-6, t-12, and t-24 time steps.
  * **Rolling Stats:** Moving averages and standard deviations (window=6).

### 3\. Model Architectures

  * **Baselines:** Linear Regression, Random Forest Regressor.
  * **RNNs:** Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU).
  * **Hybrid:** **CNN-LSTM** (1D Convolution for feature extraction feeding into LSTM for temporal sequencing).

## 🏆 Results

The models were evaluated on the unseen test set.

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **CNN-LSTM (Hybrid)** | **30.12** | **65.07** | **0.45** |
| Random Forest | 60.15 | 116.73 | -0.76 |
| GRU | 94.81 | 129.19 | -1.16 |
| LSTM | 95.12 | 129.45 | -1.17 |

**Key Finding:** The **CNN-LSTM** hybrid model significantly outperformed all other models. The Convolutional layer acted as an effective noise filter for the volatile sensor data, allowing the LSTM layer to learn sequential patterns more effectively.

## 🚀 How to Run

### Prerequisites

You can run in Google Colab or Visual Studio Code:

```bash
pip install -r requirements.txt
```

### Running the Analysis (Jupyter Notebook)

For a visual walkthrough of EDA and Training:

```bash
jupyter notebook notebooks/energy_prediction.ipynb
```

and run entire cells to get the summarized output data


## 📜 License

This project is licensed under the MIT License.
