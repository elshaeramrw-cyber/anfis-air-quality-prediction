# Air Quality & Benzene Concentration Prediction (ANFIS)

An **Adaptive Neuro-Fuzzy Inference System (ANFIS)** pipeline in MATLAB designed to estimate hazardous Benzene ($C_6H_6$) levels and air-quality risk classes from noisy, low-cost metal-oxide (MOX) gas sensor readings (UCI Air Quality dataset).

## 📂 Repository Structure

| File | Method & Focus | Description |
| :--- | :--- | :--- |
| `anfis_main_subtractive.m` | **Subtractive Clustering (`genfis2`)** | End-to-end preprocessing, sensor correlation ranking (`PT08.S1`–`PT08.S5`), and 10-run Monte Carlo ANFIS evaluation (`90.71% ± 1.14%` accuracy, `1.2950 µg/m³` RMSE). |
| `experiments/exp1_grid_partition_gauss.m` | **Grid Partition (Gaussian MFs)** | Evaluates ANFIS rule generation using Gaussian membership functions. |
| `experiments/exp2_two_sensor_baseline.m` | **Reduced Sensor Input** | Benchmarks predictive performance using a compact two-sensor subset. |
| `experiments/exp3_grid_partition_gbell.m` | **Grid Partition (Bell MFs)** | Compares generalized bell membership functions for nonlinear sensor mapping. |
| `experiments/exp4_fcm_clustering.m` | **Fuzzy C-Means (FCM)** | Explores FCM-based fuzzy rule initialization and clustering behavior. |
| `experiments/exp5_no_smoothing.m` | **Raw Signal Baseline** | Tests ANFIS modeling without temporal signal smoothing to quantify noise impact. |

## 🛠️ Tech Stack
* **Language & Toolboxes:** MATLAB, Fuzzy Logic Toolbox
