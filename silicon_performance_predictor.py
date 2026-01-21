import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def generate_silicon_dataset(samples=1200):
    """
    Simulates post-silicon characterization data.
    Models the relationship between Process, Voltage, and Temperature (PVT) 
    and hardware performance metrics.
    """
    np.random.seed(42)
    
    # Input Features (PVT & Physical variation)
    # 0: Slow, 1: Typical, 2: Fast
    process_corner = np.random.choice([0, 1, 2], samples) 
    voltage = np.random.uniform(0.65, 1.2, samples)       # Supply Voltage (V)
    temperature = np.random.uniform(-40, 125, samples)   # Junction Temp (C)
    effective_leff = np.random.normal(7, 0.2, samples)   # Effective Channel Length (nm)

    # Output Targets (Physical Models)
    # Power ~ C * V^2 * f + Leakage(temp, process)
    # We model a simplified version of this non-linear relationship
    base_power = (voltage**2 * 0.8) 
    leakage = (process_corner * 0.15) + (np.exp(temperature/100) * 0.05)
    power_watts = base_power + leakage + np.random.normal(0, 0.02, samples)
    
    # Frequency ~ (V - Vt) / L
    # Higher voltage and faster process increase frequency
    freq_ghz = (voltage * 1.5) + (process_corner * 0.4) - (temperature * 0.001) - (effective_leff * 0.05)
    freq_ghz += np.random.normal(0, 0.03, samples)

    df = pd.DataFrame({
        'process_corner': process_corner,
        'vdd_voltage': voltage,
        'temp_celsius': temperature,
        'leff_nm': effective_leff,
        'power_w': power_watts,
        'fmax_ghz': freq_ghz
    })
    
    return df

def train_and_evaluate(df):
    """
    Trains a Random Forest to predict hardware metrics and evaluates performance.
    """
    # Define Features and Targets
    X = df[['process_corner', 'vdd_voltage', 'temp_celsius', 'leff_nm']]
    y = df[['power_w', 'fmax_ghz']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit Multi-output Regressor
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Metrics
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"--- Model Analytics ---")
    print(f"R^2 Accuracy Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    return model, X_test, y_test, predictions

def plot_silicon_insights(df, y_test, predictions):
    """
    Generates engineering plots for the project portfolio.
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Vdd vs Power (Physics Validation)
    sns.scatterplot(data=df, x='vdd_voltage', y='power_w', hue='process_corner', 
                    palette='viridis', alpha=0.6, ax=ax1)
    ax1.set_title('Voltage vs. Power (Colored by Process Corner)', fontsize=14)
    ax1.set_xlabel('Supply Voltage (V)')
    ax1.set_ylabel('Total Power (W)')

    # Plot 2: Actual vs Predicted Fmax (Model Validation)
    ax2.scatter(y_test['fmax_ghz'], predictions[:, 1], alpha=0.5, color='crimson')
    ax2.plot([y_test['fmax_ghz'].min(), y_test['fmax_ghz'].max()], 
             [y_test['fmax_ghz'].min(), y_test['fmax_ghz'].max()], 
             'k--', lw=2)
    ax2.set_title('Model Performance: Actual vs. Predicted Fmax', fontsize=14)
    ax2.set_xlabel('Measured Fmax (GHz)')
    ax2.set_ylabel('Predicted Fmax (GHz)')

    plt.tight_layout()
    plt.savefig('silicon_performance_report.png')
    print("\nVisualizations saved to 'silicon_performance_report.png'")
    plt.show()

if __name__ == "__main__":
    print("Initializing Post-Silicon Data Pipeline...")
    data = generate_silicon_dataset()
    
    print("Executing Machine Learning Training...")
    model, X_test, y_test, preds = train_and_evaluate(data)
    
    print("Generating Engineering Visualizations...")
    plot_silicon_insights(data, y_test, preds)
