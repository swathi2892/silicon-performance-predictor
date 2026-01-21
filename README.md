Silicon Performance Predictor

A post-silicon characterization pipeline that models the relationship between Process, Voltage, and Temperature (PVT) and hardware performance metrics using Machine Learning.

🚀 Overview

The script simulates characterization data for silicon samples, modeling two primary hardware outputs based on semiconductor physics:

Power Consumption ($P$): Modeled with dynamic ($V^2 \cdot f$) and exponential leakage components.

Maximum Frequency ($F_{max}$): Modeled using voltage-dependent delay and thermal throttling.

📊 Data Schema

The dataset generates features representative of modern 7nm-class nodes:

Feature

Description

Range

process_corner

Manufacturing variation

0 (Slow), 1 (Typical), 2 (Fast)

vdd_voltage

Supply Voltage

0.65V to 1.2V

temp_celsius

Junction Temperature

-40°C to 125°C

leff_nm

Effective Channel Length

$\mu=7.0$ nm

power_w

Output: Total Power

Watts (W)

fmax_ghz

Output: Max Frequency

GigaHertz (GHz)

🛠️ Usage

Install Dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn


Run the Pipeline:

python silicon_performance_predictor.py


📈 Methodology

The project utilizes a RandomForestRegressor for multi-output regression. This allows the model to capture the non-linear interactions between temperature-induced leakage and voltage scaling simultaneously.

Model: Random Forest (150 estimators)

Validation: $R^2$ Score and Mean Absolute Error (MAE)

Visualization: Generates silicon_performance_report.png showing the correlation between physical silicon traits and predicted performance.

![Silicon Performance Graph](perfromance_graph.png)


📜 License

This project is for educational and engineering simulation purposes.


