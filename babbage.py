import argparse
import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import subprocess
import time
import requests
from typing import List, Dict, Tuple, Optional

# Set up logging to track operations and debug issues
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Handle imports for both package and standalone execution
# Allows running from different directory structures
try:
    from .core.simulation import TrafficSimulator
    from .core.models import ModelManager
    from .core.evaluation import ResultsAnalyzer
except ImportError:
    # If running directly as script, use local imports
    from core.simulation import TrafficSimulator
    from core.models import ModelManager
    from core.evaluation import ResultsAnalyzer


class BabbageSystem:
    def __init__(self, data_dir="data", models_dir="models", results_dir="results"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Ensure all required directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up the three main system components
        self.simulator = TrafficSimulator()
        self.model_manager = ModelManager()
        self.analyzer = ResultsAnalyzer()
        
        # Websites I'm targeting for fingerprinting classification
        # These represent different traffic patterns and use cases
        self.websites = [
            'https://www.google.com',
            'https://www.youtube.com',
            'https://www.wikipedia.org',
            'https://www.github.com',
            'https://www.stackoverflow.com'
        ]
    
    def simulate_data(self, samples_per_site=50, augment=True):
        logger.info(f"Generating synthetic data: {samples_per_site} samples per website")
        
        # Create the core synthetic traffic samples
        features_df, labels, sequences = self.simulator.generate_dataset(
            self.websites, samples_per_site
        )
        
        # Apply data augmentation to increase dataset size and robustness
        if augment:
            logger.info("Applying data augmentation...")
            features_df, labels = self.simulator.augment_data(features_df, labels)
        
        # Store the generated data for later use
        self._save_data(features_df, labels, sequences, "synthetic")
        
        logger.info(f"Generated {len(features_df)} synthetic samples")
        return features_df, labels, sequences
    
    def collect_real_data(self, samples_per_site=20, interface="wlp0s20f3"):
        if os.geteuid() != 0:
            logger.error("Real data collection requires sudo privileges")
            logger.info("Run: sudo python babbage.py --mode collect")
            return None, None, None
        
        logger.info(f"Collecting real network data on interface {interface}")
        
        all_features = []
        all_labels = []
        
        for website in self.websites:
            logger.info(f"Collecting data for {website}...")
            
            # Collect multiple samples to get statistical significance
            for sample in range(samples_per_site):
                # Capture network packets while visiting the website
                features = self._collect_website_sample(website, sample, interface)
                if features:
                    all_features.append(features)
                    all_labels.append(self._extract_label(website))
        
        if all_features:
            features_df = pd.DataFrame(all_features)
            self._save_data(features_df, all_labels, [], "real")
            logger.info(f"Collected {len(features_df)} real samples")
            return features_df, all_labels, []
        else:
            logger.error("No real data collected")
            return None, None, None
    
    def train_models(self, data_source="synthetic", optimization_level="advanced"):
        
        # Get the previously generated or collected training data
        features_df, labels = self._load_data(data_source)
        if features_df is None:
            logger.error(f"No {data_source} data found. Run simulation first.")
            return None
        
        logger.info(f"Training models on {len(features_df)} samples")
        
        # Choose training approach based on desired complexity vs performance
        if optimization_level == "basic":
            results = self.model_manager.train_basic_models(features_df, labels)
        elif optimization_level == "advanced":
            results = self.model_manager.train_advanced_models(features_df, labels)
        elif optimization_level == "ensemble":
            results = self.model_manager.train_ensemble_models(features_df, labels)
        
        # Persist trained models to disk for later use
        self._save_models(results, data_source)
        
        # Analyze results
        analysis = self.analyzer.analyze_results(results, features_df, labels)
        # Skip JSON saving for now to avoid serialization issues
        # self._save_analysis(analysis, data_source)
        
        logger.info("Model training completed")
        return results
    
    def classify_traffic(self, live=False, model_name="best"):
        
        # Load best model
        model_info = self._load_best_model(model_name)
        if not model_info:
            logger.error("No trained models found. Run training first.")
            return
        
        if live:
            logger.info("Starting live traffic classification...")
            self._classify_live_traffic(model_info)
        else:
            logger.info("Classifying test data...")
            self._classify_test_data(model_info)
    
    def benchmark_all(self):
        logger.info("Benchmarking all training approaches...")
        
        results = {}
        
        # 1. Basic synthetic data
        logger.info("1. Testing basic synthetic data...")
        features_df, labels, _ = self.simulate_data(samples_per_site=30, augment=False)
        basic_results = self.model_manager.train_basic_models(features_df, labels)
        results['basic_synthetic'] = self._get_best_accuracy(basic_results)
        
        # 2. Augmented synthetic data
        logger.info("2. Testing augmented synthetic data...")
        features_df, labels, _ = self.simulate_data(samples_per_site=50, augment=True)
        aug_results = self.model_manager.train_advanced_models(features_df, labels)
        results['augmented_synthetic'] = self._get_best_accuracy(aug_results)
        
        # 3. Ensemble models
        logger.info("3. Testing ensemble models...")
        ensemble_results = self.model_manager.train_ensemble_models(features_df, labels)
        results['ensemble'] = self._get_best_accuracy(ensemble_results)
        
        # Display benchmark results
        self._display_benchmark_results(results)
        
        return results
    
    def _collect_website_sample(self, website, sample_id, interface):
        pcap_file = self.data_dir / f"temp_{sample_id}.pcap"
        
        # Start packet capture
        capture_process = subprocess.Popen([
            'sudo', 'tcpdump', '-i', interface,
            '-w', str(pcap_file),
            f'host {website.replace("https://", "").replace("www.", "")}'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(2)  # Let capture start
        try:
            # Visit website
            response = requests.get(website, timeout=10, verify=True)
            response.raise_for_status()
            time.sleep(8)  # Let traffic flow
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error visiting {website}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error visiting {website}: {e}")
        except Exception as e:
            logger.warning(f"Error visiting {website}: {e}")
        
        # Stop capture
        capture_process.terminate()
        time.sleep(1)
        
        # Extract features
        if pcap_file.exists() and pcap_file.stat().st_size > 0:
            features = self.simulator.extract_features_from_pcap(pcap_file)
            pcap_file.unlink()  # Clean up
            return features
        
        return None
    
    def _save_data(self, features_df, labels, sequences, data_type):
        features_df.to_csv(self.data_dir / f"{data_type}_features.csv", index=False)
        
        with open(self.data_dir / f"{data_type}_labels.txt", 'w') as f:
            for label in labels:
                f.write(f"{label}\n")
        
        if sequences:
            with open(self.data_dir / f"{data_type}_sequences.pkl", 'wb') as f:
                pickle.dump(sequences, f)
    
    def _load_data(self, data_type):
        features_file = self.data_dir / f"{data_type}_features.csv"
        labels_file = self.data_dir / f"{data_type}_labels.txt"
        
        if not features_file.exists():
            return None, None
        
        features_df = pd.read_csv(features_file)
        
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f]
        
        return features_df, labels
    
    def _save_models(self, results, data_source):
        for model_name, model_info in results.items():
            model_file = self.models_dir / f"{data_source}_{model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_info, f)
    
    def _save_analysis(self, analysis, data_source):
        analysis_file = self.results_dir / f"{data_source}_analysis.json"
        import json
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def _load_best_model(self, model_name):
        # Find best model file
        model_files = list(self.models_dir.glob("*.pkl"))
        if not model_files:
            return None
        
        # For now, return the most recent one
        best_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        
        with open(best_model_file, 'rb') as f:
            return pickle.load(f)
    
    def _classify_live_traffic(self, model_info):
        logger.info("Live classification not yet implemented")
        logger.info("This would capture live traffic and classify in real-time")
    
    def _classify_test_data(self, model_info):
        logger.info("Test data classification not yet implemented")
        logger.info("This would classify a test dataset")
    
    def _get_best_accuracy(self, results):
        if not results:
            return 0.0
        
        best_acc = 0.0
        for model_name, model_info in results.items():
            if 'accuracy' in model_info:
                best_acc = max(best_acc, model_info['accuracy'])
        
        return best_acc
    
    def _display_benchmark_results(self, results):
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        
        for approach, accuracy in results.items():
            print(f"{approach:20s}: {accuracy:.1%}")
        
        best_approach = max(results.keys(), key=lambda k: results[k])
        print(f"\nBest approach: {best_approach} ({results[best_approach]:.1%})")
    
    def _extract_label(self, website_url):
        return website_url.replace('https://www.', '').replace('.com', '')


def main():
    parser = argparse.ArgumentParser(description='Babbage Traffic Fingerprinting System')
    parser.add_argument('--mode', required=True, 
                       choices=['train', 'simulate', 'collect', 'classify', 'benchmark'],
                       help='Operation mode')
    parser.add_argument('--samples', type=int, default=50,
                       help='Samples per website (default: 50)')
    parser.add_argument('--optimization', choices=['basic', 'advanced', 'ensemble'],
                       default='advanced', help='Training optimization level')
    parser.add_argument('--data-source', choices=['synthetic', 'real'],
                       default='synthetic', help='Data source for training')
    parser.add_argument('--live', action='store_true',
                       help='Enable live classification')
    parser.add_argument('--interface', default='wlp0s20f3',
                       help='Network interface for data collection')
    
    args = parser.parse_args()
    
    # Initialize system
    babbage = BabbageSystem()
    
    try:
        if args.mode == 'simulate':
            logger.info("Mode: Data Simulation")
            features_df, labels, sequences = babbage.simulate_data(
                samples_per_site=args.samples, augment=True
            )
            if features_df is not None:
                logger.info("Training models on simulated data...")
                babbage.train_models('synthetic', args.optimization)
        
        elif args.mode == 'train':
            logger.info("Mode: Model Training")
            babbage.train_models(args.data_source, args.optimization)
        
        elif args.mode == 'collect':
            logger.info("Mode: Real Data Collection")
            features_df, labels, sequences = babbage.collect_real_data(
                samples_per_site=args.samples, interface=args.interface
            )
            if features_df is not None:
                logger.info("Training models on real data...")
                babbage.train_models('real', args.optimization)
        
        elif args.mode == 'classify':
            logger.info("Mode: Traffic Classification")
            babbage.classify_traffic(live=args.live)
        
        elif args.mode == 'benchmark':
            logger.info("Mode: Benchmark All Approaches")
            babbage.benchmark_all()
        
        logger.info("Operation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()