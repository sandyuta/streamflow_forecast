# Comparative Analysis of RNN Architectures for Daily Streamflow Forecasting

## Overview
Accurate streamflow forecasting is crucial for water resource management, particularly in reservoir management and flood warning systems. Traditional physics-based models require manual calibration and often struggle with snowpack due to the complex underlying physics. This project applies Deep Learning (RNN, LSTM) and classical Machine Learning algorithms to predict streamflow. The primary goal is to assess how well these models capture hydrologic memory in two distinct types of hydrological regimes: a snow-driven catchment in the Colorado Rockies and a rainfall-driven catchment in the Southeast US.

## Objectives
* Predict next-day daily streamflow using meteorological variables and historical streamflow.
* Compare Deep Learning models against classical time-series and tree-based baselines.
* Evaluate the models' ability to capture hydrologic memory in rainfall-driven vs. snowfall-driven catchments.

## Repository Structure
* `data/`: Contains dataset details (note: actual large datasets are not uploaded here).
* `src/`: Source code for data processing, baseline models, deep learning models, and evaluation.
* `outputs/`: Generated figures, tables, and model artifacts.
* `docs/`: Related literature, final reports, and presentations.
* `requirements.txt`: Python dependencies required to run the project.

## Data
The project utilizes the CAMELS (Catchment Attributes and Meteorology for Large-sample Studies) dataset, which contains meteorological variables and streamflow data from USGS stations.
* **Source:** CAMELS dataset (prepackaged)
* **Features:** Time series of precipitation, maximum temperature, minimum temperature, and previous day streamflow.
* **Label:** Next day streamflow.
* **Period:** 15 years of continuous daily data (10 years training, 3 years validation/hyperparameter tuning, 3 years testing).
* **Included in Repo:** Due to size constraints, the raw data is not included. Please see `data/README.md` for instructions on how to download and place the dataset.

## Methods
We predict streamflow using multiple approaches to establish a comprehensive benchmark:
1. **Classical & Tree-based Baselines:** Multiple Linear Regression (MLR), Autoregressive Models with Exogenous Inputs (ARX), Random Forest, and XGBoost (using lagged features).
2. **Deep Learning:** Basic Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks.
* **Evaluation Metrics:** Nash-Sutcliffe Efficiency (NSE) and Root Mean Square Error (RMSE).

## Requirements
* Python 3.8+
* See `requirements.txt` for the full list of required packages.

## How to Run the Project
*(Instructions will be added here as the code is developed)*

## Authors
* Sandesh Adhikari (ID: 1002127398)

## Acknowledgments
* Dr. Jinzhu Yu
* CAMELS dataset providers.