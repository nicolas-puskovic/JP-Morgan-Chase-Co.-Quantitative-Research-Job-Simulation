# JP-Morgan-Chase-Co.-Quantitative-Research-Job-Simulation
<img width="2505" height="1214" alt="image" src="https://github.com/user-attachments/assets/6e6dd70d-f51d-48be-a60a-aabfc8763f71" />
## 💡 Methodological Evolution: Why SARIMA over Linear Regression?

This project was initially inspired by the J.P. Morgan Quantitative Research task. The standard solution typically suggests a **Linear Regression** combined with a fixed **Fourier Transform** (Sine/Cosine waves) to model price and seasonality.

While functional for basic estimation, that approach has significant limitations in a production environment. I chose to re-engineer the core logic using a **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** framework.

### Comparative Analysis

| Feature | Standard Approach (Linear Reg + Fourier) | My Solution (SARIMA) |
| :--- | :--- | :--- |
| **Trend Modeling** | **Deterministic**: Assumes an infinite linear growth. Prone to significant error if the structural trend changes. | **Stochastic**: Captures the trend dynamically through differencing ($d$), adapting to recent market shifts. |
| **Seasonality** | **Rigid**: Fits a perfect sine wave. Fails if winter peaks are sharper than summer troughs. | **Adaptive**: Learns the specific seasonal structure ($P, D, Q)_{12}$ from the data, capturing complex periodic patterns. |
| **Overfitting Risk** | **High**: The Fourier terms can force-fit noise as signal ("learning the noise"). | **Controlled**: Mitigated via AIC (Akaike Information Criterion) selection and residual analysis. |
| **Visualization** | **Static**: A simple `.png` plot. | **Interactive**: A Plotly Dashboard allowing drill-down analysis and zoom. |

### Validation Result
By switching to SARIMA, the model achieved a **MAPE < 2%** on out-of-sample data (last 6 months hidden), confirmed by a successful **Ljung-Box test** ($p > 0.05$) indicating residuals are White Noise.
