# AI-Driven Energy Consumption Prediction in Smart Homes

> **Research paper:** "AI-Driven Energy Consumption Prediction in Smart Homes using IoT and Edge Computing"

This repository implements a complete, reproducible research pipeline for short-term household energy consumption prediction using deep learning, edge computing, and federated learning.

## Features

| Component | Implementation |
|---|---|
| **Data Preprocessing** | Modular pipeline with chronological splitting, MinMax normalisation |
| **Baseline Models** | ARIMA, Standard LSTM, CNN-LSTM |
| **Proposed Models** | BiLSTM, LEF-BiLSTM (Lightweight Edge-Friendly) |
| **Edge Computing** | TensorFlow Lite conversion, IoT edge gateway simulation |
| **MQTT IoT Simulation** | paho-mqtt based sensor data streaming |
| **Federated Learning** | Manual FedAvg across 3 simulated clients |
| **Evaluation** | MAE, RMSE, R² with automated comparison tables |
| **Visualisation** | Publication-quality figures for all metrics |

## Dataset

- **Source:** [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Location:** Single household in Sceaux, France
- **Period:** December 2006 – November 2010
- **Resolution:** 1 reading per minute (~2.07 million rows)
- **Used:** First 20,000 rows (configurable in `config/settings.py`)
- **Target:** `Global_active_power` (kW) — univariate forecasting

### Setup

1. Download the dataset from the UCI link above
2. Place the file as `dataset/data.txt` in the project root

## Project Structure

```
├── config/                  # Centralized configuration
│   ├── __init__.py
│   └── settings.py          # All hyperparameters, paths, constants
├── dataset/                 # Dataset directory
│   └── data.txt             # Raw dataset (not tracked in git)
├── preprocessing/           # Data loading and feature engineering
│   ├── __init__.py
│   └── pipeline.py          # Full preprocessing pipeline
├── models/                  # All model architectures
│   ├── __init__.py
│   ├── base.py              # Abstract base model interface
│   ├── keras_base.py        # Shared Keras model mixin
│   ├── registry.py          # Model name → class registry
│   ├── arima_model.py       # ARIMA baseline
│   ├── lstm_model.py        # Standard LSTM
│   ├── cnn_lstm_model.py    # CNN-LSTM hybrid
│   ├── bilstm_model.py      # Bidirectional LSTM
│   └── lef_bilstm_model.py  # LEF-BiLSTM (edge-friendly)
├── training/                # Training utilities
│   ├── __init__.py
│   └── trainer.py           # Reusable training driver
├── evaluation/              # Evaluation and comparison
│   ├── __init__.py
│   ├── evaluator.py         # Metrics, predictions export
│   ├── comparison.py        # Multi-model baseline comparison
│   └── visualisation.py     # Publication-quality figures
├── edge/                    # Edge computing pipeline
│   ├── __init__.py
│   ├── tflite_converter.py  # Keras → TFLite conversion
│   ├── edge_gateway.py      # Simulated IoT edge gateway
│   ├── latency.py           # Latency benchmarking
│   └── mqtt_simulator.py    # MQTT IoT simulation
├── federated/               # Federated learning
│   ├── __init__.py
│   └── fedavg.py            # FedAvg simulation (no TFF)
├── figures/                 # Generated figures (not tracked)
├── outputs/                 # Generated outputs (not tracked)
│   ├── metrics/             # CSV files, summaries
│   ├── models/              # Saved model checkpoints
│   ├── tflite/              # TFLite models
│   └── logs/                # Training logs
├── reports/                 # Documentation
│   ├── dataset_info.md
│   ├── Implementation_Report.md
│   └── Explanation_Report.md
├── main.py                  # Entry point — runs full pipeline
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Place dataset
# Download from UCI and save as dataset/data.txt

# 4. Run the full pipeline
python main.py
```

## Generated Outputs

| File | Description |
|---|---|
| `outputs/metrics/metrics.csv` | MAE, RMSE, R² for all models |
| `outputs/metrics/comparison.csv` | Full baseline comparison table |
| `outputs/metrics/predictions.csv` | Per-model actual vs predicted |
| `outputs/metrics/training_history.csv` | Per-epoch loss and LR |
| `outputs/metrics/latency.csv` | Keras vs TFLite latency |
| `outputs/metrics/federated_log.csv` | FedAvg convergence log |
| `outputs/metrics/hyperparameters.txt` | All hyperparameter values |
| `outputs/metrics/*_model_summary.txt` | Model architecture summaries |
| `outputs/tflite/model.tflite` | TFLite model for edge deployment |
| `outputs/logs/training.log` | Full execution log |
| `figures/*.png` | Publication-quality comparison figures |

## Configuration

All hyperparameters are centralized in `config/settings.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `NUM_ROWS` | 20,000 | Number of dataset rows to use |
| `TIME_STEP` | 30 | Sliding window size (minutes) |
| `EPOCHS` | 30 | Maximum training epochs |
| `BATCH_SIZE` | 32 | Training batch size |
| `LSTM_UNITS` | 32 | LSTM hidden units |
| `NUM_CLIENTS` | 3 | Federated learning clients |
| `NUM_FED_ROUNDS` | 5 | Federated learning rounds |

## Models

### 1. ARIMA
Statistical baseline using autoregressive integrated moving average.

### 2. Standard LSTM
Two-layer unidirectional LSTM with dropout regularization.

### 3. CNN-LSTM
Conv1D feature extraction followed by LSTM sequence modelling.

### 4. BiLSTM
Two-layer Bidirectional LSTM — processes sequences in both directions.

### 5. LEF-BiLSTM (Proposed)
Lightweight Edge-Friendly BiLSTM designed for edge/IoT deployment:
- Single BiLSTM layer with halved units (16 vs 32)
- L2 kernel regularization
- ~4× fewer parameters than full BiLSTM
- Optimized for TFLite conversion

## Edge Computing

The edge pipeline demonstrates on-device inference:
1. **TFLite Conversion**: Keras model → quantized TFLite format
2. **Edge Gateway**: Simulated IoT gateway with sliding-window inference
3. **MQTT Simulation**: Sensor data publishing via MQTT protocol
4. **Latency Benchmarking**: Keras vs TFLite inference speed comparison

## Federated Learning

Manual FedAvg implementation (no TensorFlow Federated):
1. Training data partitioned across 3 simulated clients
2. Each client trains locally for 3 epochs per round
3. Server averages client weights (Federated Averaging)
4. Global model evaluated on validation set after each round
5. Process repeats for 5 rounds

## License

This project is for academic/research purposes.
