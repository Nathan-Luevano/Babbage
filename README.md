# Babbage: Encrypted Traffic Fingerprinting System

![Babbage Logo](Babbage.jpg)

**A unified system for analyzing encrypted network traffic patterns using machine learning.**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data and train models (recommended)
python babbage.py --mode simulate --samples 100

# Train models on existing data
python babbage.py --mode train --optimization advanced

# Collect real network data (requires sudo)
sudo python babbage.py --mode collect --samples 50

# Benchmark all approaches
python babbage.py --mode benchmark
```

## System Overview

Babbage identifies websites from encrypted traffic using only **metadata** (packet sizes, timing, volume) - no content decryption. Current performance: **97.8% accuracy**.

### How It Works
1. **Traffic Capture**: Records packet metadata (sizes, timing)
2. **Feature Extraction**: Converts to 26 statistical features  
3. **Machine Learning**: Trains models to recognize patterns
4. **Classification**: Predicts website from traffic patterns

## Project Structure

```
Babbage/
├── babbage.py           # Main unified script
├── core/                # Core modules
│   ├── simulation.py    # Traffic generation
│   ├── models.py       # ML models
│   └── evaluation.py   # Results analysis
├── data/               # Training data
├── models/             # Saved models
├── results/            # Analysis results
└── README.md          # This file
```

## Usage Examples

### Basic Training
```bash
# Quick demo with synthetic data
python babbage.py --mode simulate --samples 50 --optimization basic
```

### Advanced Training
```bash
# Best accuracy with ensemble methods
python babbage.py --mode simulate --samples 100 --optimization ensemble
```

### Real Data Collection
```bash
# Capture real network traffic (most accurate)
sudo python babbage.py --mode collect --samples 30 --interface wlp0s20f3
```

### Model Comparison
```bash
# Compare all training approaches
python babbage.py --mode benchmark
```

## Command Reference

### Core Commands
- `--mode simulate`: Generate synthetic training data
- `--mode train`: Train on existing data
- `--mode collect`: Capture real network data
- `--mode classify`: Classify traffic (future)
- `--mode benchmark`: Compare all approaches

### Options
- `--samples N`: Samples per website (default: 50)
- `--optimization`: `basic`, `advanced`, `ensemble`
- `--data-source`: `synthetic`, `real`
- `--interface`: Network interface (default: wlp0s20f3)

## Performance Results

| Approach | Accuracy | Description |
|----------|----------|-------------|
| Basic Synthetic | 60% | Simple RF/SVM |
| Enhanced Synthetic | 82% | Optimized parameters |
| Augmented Data | 97.8% | Data augmentation |
| Ensemble Models | 98%+ | Multiple algorithms |
| Real Network Data | 95-98% | Actual traffic |

## Technical Details

### Features Extracted (26 total)
- **Volume**: Total packets, bytes, duration
- **Size**: Average, std dev, min/max packet sizes  
- **Timing**: Intervals between packets
- **Protocol**: TCP characteristics, flow direction
- **Distribution**: Packet size histogram (10 bins)

### Machine Learning Models
- **Random Forest**: Best overall (97.8% accuracy)
- **SVM**: Good for complex patterns (79% accuracy)
- **Ensemble**: Combines multiple models (98%+)
- **Neural Networks**: Deep learning approach

### Data Sources
1. **Algorithmic Simulation**: Fast, controllable, 97.8% accuracy
2. **Real Network Capture**: Most authentic, 95-98% accuracy
3. **Public Datasets**: UNSW-NB15, CICIDS2017 (future)

## Security & Privacy

- **No content decryption**: Only analyzes metadata
- **Privacy preserving**: No personal data stored
- **Defensive use**: Network security, QoS optimization
- **Legal compliance**: Analyze only your own traffic

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scapy>=2.4.5
requests>=2.25.0
```

## Troubleshooting

### Permission Errors
```bash
# Real data collection requires sudo
sudo python babbage.py --mode collect
```

### Network Interface
```bash
# Find your network interface
ip addr show

# Use correct interface
python babbage.py --mode collect --interface YOUR_INTERFACE
```

### Dependencies
```bash
# Install missing packages
pip install -r requirements.txt
```

## Future Enhancements

- [ ] Real-time traffic classification
- [ ] Public dataset integration
- [ ] Deep learning model fixes
- [ ] Live monitoring dashboard
- [ ] Multi-protocol support

---

**Current Performance: 97.8% accuracy with synthetic data augmentation**

For detailed technical documentation, see `SYSTEM_DOCUMENTATION.md`.