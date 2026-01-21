# silicon-performance-predictor
A Machine Learning framework for predicting post-silicon power and timing metrics based on PVT variations.

🚀 Overview

This repository contains a high-fidelity simulator and predictive engine that bridges Hardware Engineering and Data Science. The tool models the relationship between semiconductor physical parameters—Process, Voltage, and Temperature (PVT)—and critical output metrics like Power Consumption and Maximum Clock Frequency ($F_{max}$).

In modern SoC (System on Chip) design, understanding silicon variation is essential for product binning, yield optimization, and power management.

🛠️ Technical Implementation

Synthetic Data Generation: Models complex physics, including quadratic dynamic power scaling ($P \propto V^2f$) and exponential leakage dependencies.

Machine Learning Architecture: Utilizes a Multi-output Random Forest Regressor to capture non-linear interactions between supply voltage and thermal effects.

Analytics Pipeline: Features a full validation suite that calculates $R^2$ scores and Mean Absolute Error (MAE) for hardware validation.

📊 Key Insights

The model analyzes how the following variables impact the chip:

Process Corner: Variations in transistor manufacturing (Slow, Typical, Fast).

Voltage ($V_{dd}$): The supply voltage ranging from 0.65V to 1.2V.

Temperature ($T_j$): Junction temperatures from -40°C to 125°C.

Effective Leff: Sub-nanometer channel length variations.

💻 Usage

Dependencies:

pip install pandas scikit-learn matplotlib seaborn


Execution:

python silicon_performance_predictor.py


📈 Portfolio Visuals

The script automatically generates a silicon_performance_report.png file, which includes:
## Performance Analysis
![Silicon Performance Graph](perfromance_graph.png)

Voltage vs. Power Analysis: A visualization of the power envelope across different process corners.

Model Accuracy Plot: A regression plot comparing predicted vs. measured $F_{max}$, demonstrating the model's reliability.

💡 Industrial Applications

Speed Binning: Automating the categorization of chips into different performance tiers.

Adaptive Voltage Scaling (AVS): Real-time power optimization based on sensor data.

Yield Analysis: Predicting performance failures before chips reach final testing.

Developed as a demonstration of applying Machine Learning to VLSI and Computer Architecture domains.
