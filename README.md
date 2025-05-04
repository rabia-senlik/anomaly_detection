
# Anomaly Detection Project

## Project Overview
This project focuses on anomaly detection using various machine learning techniques. It uses a dataset related to the MetroPT3 system, and performs data analysis, anomaly detection, and visualization. Additionally, a web interface is built using Streamlit for easier interaction and results display.

## Directory Structure
```
├── anomaly
│   ├── analysis.py               # Anomalies analysis related to the detection process
│   ├── compare_models.py         # Compares the performance of different models
│   ├── isolation_forest.py       # Implements the Isolation Forest algorithm for anomaly detection
│   ├── model_comparison.png      # Visual comparing different models
│   ├── monthly_anomaly_distribution.png  # Monthly anomaly distribution graph
│   └── tp2_anomalies.png         # Detected anomalies visualization
├── data
│   ├── Data Description_Metro.pdf  # Description of the MetroPT3 dataset
│   └── MetroPT3.csv              # The actual MetroPT3 dataset
├── data_loader
│   └── data.py                   # The data loading script for preparing the dataset
├── eda
│   ├── sensor_visuals.png        # Sensor data visualizations
│   └── visuals.py                # Visualizations for exploratory data analysis (EDA)
└── streamlit_app
    └── app.py                    # Streamlit-based app for displaying results
```

## Requirements
To run the project, ensure you have the following dependencies installed:

- Python 3.x
- `pip install -r requirements.txt`

The `requirements.txt` file will include necessary libraries like Pandas, Scikit-learn, Matplotlib, and Streamlit.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scriptsctivate'
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Data Loading:
Run the following script to load the dataset:

```bash
python data_loader/data.py
```

### 2. Anomaly Detection:
Run the anomaly detection models:

```bash
python anomaly/isolation_forest.py
```

You can also compare different models by using:

```bash
python anomaly/compare_models.py
```

### 3. EDA and Visualization:
To generate exploratory data analysis visuals, run:

```bash
python eda/visuals.py
```

### 4. Streamlit App:
To run the Streamlit app and view the anomaly detection results interactively, use:

```bash
streamlit run streamlit_app/app.py
```

This will open a local web interface where you can interact with the results.


