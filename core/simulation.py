import numpy as np
import pandas as pd
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from scapy.all import rdpcap, IP, TCP

logger = logging.getLogger(__name__)


class TrafficSimulator:
    def __init__(self):
        self.websites_profiles = self._initialize_profiles()
    
    def _initialize_profiles(self):
        return {
            'google': {
                'packet_count_range': [60, 120],
                'byte_count_range': [25000, 80000],
                'avg_packet_size_range': [400, 650],
                'interval_timing_range': [0.02, 0.15],
                'burst_probability': 0.4,
                'connection_ratio': 0.6
            },
            'youtube': {
                'packet_count_range': [150, 400],
                'byte_count_range': [100000, 500000],
                'avg_packet_size_range': [800, 1400],
                'interval_timing_range': [0.01, 0.05],
                'burst_probability': 0.8,
                'connection_ratio': 0.9
            },
            'wikipedia': {
                'packet_count_range': [40, 100],
                'byte_count_range': [15000, 60000],
                'avg_packet_size_range': [300, 600],
                'interval_timing_range': [0.05, 0.3],
                'burst_probability': 0.3,
                'connection_ratio': 0.5
            },
            'github': {
                'packet_count_range': [80, 200],
                'byte_count_range': [30000, 120000],
                'avg_packet_size_range': [450, 800],
                'interval_timing_range': [0.03, 0.2],
                'burst_probability': 0.5,
                'connection_ratio': 0.7
            },
            'stackoverflow': {
                'packet_count_range': [50, 150],
                'byte_count_range': [20000, 90000],
                'avg_packet_size_range': [350, 700],
                'interval_timing_range': [0.04, 0.25],
                'burst_probability': 0.4,
                'connection_ratio': 0.6
            }
        }
    
    def generate_dataset(self, websites: List[str], samples_per_site: int = 50):
        logger.info(f"Generating dataset for {len(websites)} websites")
        
        all_features = []
        all_labels = []
        all_sequences = []
        
        for website_url in websites:
            website_type = self._extract_website_type(website_url)
            logger.info(f"Generating {samples_per_site} samples for {website_type}")
            
            # Generate multiple unique sessions to capture traffic variability
            for session_id in range(samples_per_site):
                features = self._generate_realistic_features(website_type, session_id)
                sequences = self._generate_packet_sequences(website_type, session_id)
                
                all_features.append(features)
                all_labels.append(website_type)
                all_sequences.append(sequences)
        
        features_df = pd.DataFrame(all_features)
        logger.info(f"Generated dataset: {len(features_df)} samples, {len(features_df.columns)} features")
        
        return features_df, all_labels, all_sequences
    
    def augment_data(self, features_df: pd.DataFrame, labels: List[str]):
        logger.info("Applying data augmentation...")
        
        augmented_features = []
        augmented_labels = []
        
        # Start with the base dataset
        augmented_features.append(features_df)
        augmented_labels.extend(labels)
        
        # Add small random variations to simulate measurement noise
        noisy_df = self._add_noise(features_df, noise_level=0.1)
        augmented_features.append(noisy_df)
        augmented_labels.extend(labels)
        
        # Simulate network timing variations due to congestion, routing changes
        time_shifted_df = self._time_shift(features_df, shift_range=0.15)
        augmented_features.append(time_shifted_df)
        augmented_labels.extend(labels)
        
        # Account for different connection speeds and compression levels
        scaled_df = self._scale_features(features_df, scale_range=0.2)
        augmented_features.append(scaled_df)
        augmented_labels.extend(labels)
        
        # Merge all augmented versions into one large training set
        combined_df = pd.concat(augmented_features, ignore_index=True)
        
        logger.info(f"Augmented dataset: {len(combined_df)} samples (was {len(features_df)})")
        return combined_df, augmented_labels
    
    def extract_features_from_pcap(self, pcap_file):
        try:
            packets = rdpcap(str(pcap_file))
            
            if len(packets) < 5:
                return None
            
            packet_data = []
            for packet in packets:
                if IP in packet and TCP in packet:
                    packet_info = {
                        'size': len(packet),
                        'timestamp': float(packet.time),
                        'direction': 'outgoing' if packet[IP].src.startswith('192.168') else 'incoming',
                        'tcp_flags': packet[TCP].flags,
                        'window_size': packet[TCP].window
                    }
                    packet_data.append(packet_info)
            
            if len(packet_data) < 5:
                return None
            
            return self._extract_statistical_features(packet_data)
            
        except Exception as e:
            logger.warning(f"Error processing {pcap_file}: {e}")
            return None
    
    def _generate_realistic_features(self, website_type: str, session_id: int):
        profile = self.websites_profiles.get(website_type, self.websites_profiles['google'])
        
        # Use deterministic randomization based on session ID for reproducibility
        session_seed = hash(f"{website_type}_{session_id}") % (2**32)
        np.random.seed(session_seed)
        
        # Extract base parameters from the website's traffic profile
        packet_range = profile['packet_count_range']
        byte_range = profile['byte_count_range']
        size_range = profile['avg_packet_size_range']
        interval_range = profile['interval_timing_range']
        burst_prob = profile['burst_probability']
        conn_ratio = profile['connection_ratio']
        
        total_packets = np.random.randint(packet_range[0], packet_range[1] + 1)
        total_bytes = np.random.randint(byte_range[0], byte_range[1] + 1)
        
        # Packet size distribution
        base_avg_size = np.random.uniform(size_range[0], size_range[1])
        size_std = base_avg_size * 0.4 * np.random.uniform(0.5, 1.5)
        
        # Apply burst behavior
        if np.random.random() < burst_prob:
            burst_factor = np.random.uniform(1.3, 2.0)
            base_avg_size *= burst_factor
            size_std *= burst_factor
        
        # Timing characteristics
        base_interval = np.random.uniform(interval_range[0], interval_range[1])
        interval_std = base_interval * np.random.uniform(0.3, 1.2)
        
        # Derived metrics
        session_duration = total_packets * base_interval * np.random.uniform(0.8, 1.2)
        session_duration = max(5.0, min(180.0, session_duration))
        
        min_packet_size = max(40, int(base_avg_size * 0.1))
        max_packet_size = min(1500, int(base_avg_size * 2.0))
        median_packet_size = base_avg_size * np.random.uniform(0.9, 1.1)
        
        tcp_packets = int(total_packets * np.random.uniform(0.85, 0.98))
        outgoing_ratio = conn_ratio + np.random.normal(0, 0.08)
        outgoing_ratio = max(0.1, min(0.9, outgoing_ratio))
        
        min_interval = max(0.001, base_interval * 0.1)
        max_interval = base_interval * np.random.uniform(3.0, 10.0)
        
        # Size histogram
        size_histogram = self._generate_size_histogram(base_avg_size, size_std, total_packets)
        
        # Compile features
        features = {
            'total_packets': total_packets,
            'total_bytes': total_bytes,
            'avg_packet_size': base_avg_size,
            'std_packet_size': size_std,
            'min_packet_size': min_packet_size,
            'max_packet_size': max_packet_size,
            'median_packet_size': median_packet_size,
            'session_duration': session_duration,
            'avg_interval': base_interval,
            'std_interval': interval_std,
            'min_interval': min_interval,
            'max_interval': max_interval,
            'tcp_packets': tcp_packets,
            'outgoing_ratio': outgoing_ratio,
            'incoming_ratio': 1.0 - outgoing_ratio
        }
        
        # Add histogram bins
        for i, count in enumerate(size_histogram):
            features[f'size_bin_{i}'] = count
        
        return features
    
    def _generate_packet_sequences(self, website_type: str, session_id: int, max_length: int = 100):
        profile = self.websites_profiles.get(website_type, self.websites_profiles['google'])
        
        session_seed = hash(f"{website_type}_{session_id}_seq") % (2**32)
        np.random.seed(session_seed)
        
        size_range = profile['avg_packet_size_range']
        interval_range = profile['interval_timing_range']
        burst_prob = profile['burst_probability']
        
        seq_length = np.random.randint(max_length // 2, max_length)
        
        # Generate size sequence
        sequence = self._generate_realistic_sequence(size_range, burst_prob, seq_length)
        
        # Generate timing sequence
        base_interval = np.random.uniform(interval_range[0], interval_range[1])
        timing_sequence = self._generate_timing_sequence(base_interval, burst_prob, seq_length - 1)
        
        # Pad sequences
        if len(sequence) < max_length:
            sequence = np.pad(sequence, (0, max_length - len(sequence)), mode='constant')
        
        if len(timing_sequence) < max_length - 1:
            padding = (max_length - 1) - len(timing_sequence)
            timing_sequence = np.pad(timing_sequence, (0, padding), mode='constant')
        
        return {
            'size_sequence': sequence[:max_length].tolist(),
            'timing_sequence': timing_sequence[:max_length-1].tolist()
        }
    
    def _generate_size_histogram(self, avg_size: float, std_size: float, total_packets: int):
        sizes = []
        
        # Small packets (control/ACK)
        small_count = int(total_packets * 0.2)
        small_sizes = np.random.gamma(2, 25, small_count)
        sizes.extend(small_sizes)
        
        # Medium packets (main content)
        medium_count = int(total_packets * 0.6)
        medium_sizes = np.random.normal(avg_size, std_size, medium_count)
        sizes.extend(medium_sizes)
        
        # Large packets (data transfer)
        large_count = total_packets - small_count - medium_count
        if large_count > 0:
            large_sizes = np.random.normal(avg_size * 1.3, std_size * 0.7, large_count)
            sizes.extend(large_sizes)
        
        # Clip to realistic bounds
        sizes = np.clip(sizes, 40, 1500)
        
        # Create histogram
        hist, _ = np.histogram(sizes, bins=10, range=(0, 1500))
        return hist.tolist()
    
    def _generate_realistic_sequence(self, size_range: List[float], burst_prob: float, length: int):
        sequence = []
        i = 0
        
        while i < length:
            if np.random.random() < burst_prob and i < length - 3:
                # Burst pattern
                burst_length = min(np.random.randint(3, 6), length - i)
                burst_size = np.random.uniform(size_range[0] * 1.2, size_range[1] * 1.5)
                
                for _ in range(burst_length):
                    size = burst_size + np.random.normal(0, burst_size * 0.2)
                    sequence.append(max(40, min(1500, size)))
                
                i += burst_length
            else:
                # Normal packet
                base_size = np.random.uniform(size_range[0], size_range[1])
                size = base_size + np.random.normal(0, base_size * 0.3)
                sequence.append(max(40, min(1500, size)))
                i += 1
        
        return np.array(sequence)
    
    def _generate_timing_sequence(self, base_interval: float, burst_prob: float, length: int):
        intervals = []
        
        for _ in range(length):
            if np.random.random() < burst_prob:
                interval = base_interval * np.random.uniform(0.2, 0.6)
            elif np.random.random() < 0.15:
                interval = base_interval * np.random.uniform(2.0, 4.0)
            else:
                interval = base_interval * np.random.uniform(0.6, 1.8)
            
            intervals.append(max(0.001, interval))
        
        return np.array(intervals)
    
    def _add_noise(self, features_df: pd.DataFrame, noise_level: float = 0.1):
        noisy_df = features_df.copy()
        
        for col in features_df.select_dtypes(include=[np.number]).columns:
            noise = np.random.normal(0, noise_level * features_df[col].std(), len(features_df))
            noisy_df[col] = np.maximum(0, features_df[col] + noise)
        
        return noisy_df
    
    def _time_shift(self, features_df: pd.DataFrame, shift_range: float = 0.15):
        shifted_df = features_df.copy()
        timing_features = ['avg_interval', 'std_interval', 'min_interval', 'max_interval']
        
        shift_factor = 1 + np.random.uniform(-shift_range, shift_range, len(features_df))
        
        for feature in timing_features:
            if feature in shifted_df.columns:
                shifted_df[feature] *= shift_factor
        
        return shifted_df
    
    def _scale_features(self, features_df: pd.DataFrame, scale_range: float = 0.2):
        scaled_df = features_df.copy()
        size_features = ['total_bytes', 'avg_packet_size', 'std_packet_size', 
                        'min_packet_size', 'max_packet_size', 'median_packet_size']
        
        scale_factor = 1 + np.random.uniform(-scale_range, scale_range, len(features_df))
        
        for feature in size_features:
            if feature in scaled_df.columns:
                scaled_df[feature] = np.maximum(1, scaled_df[feature] * scale_factor)
        
        return scaled_df
    
    def _extract_statistical_features(self, packet_data: List[Dict]):
        if len(packet_data) < 5:
            return None
        
        sizes = [p['size'] for p in packet_data]
        timestamps = [p['timestamp'] for p in packet_data]
        
        # Calculate intervals
        intervals = np.diff(timestamps)
        
        # Basic statistics
        features = {
            'total_packets': len(packet_data),
            'total_bytes': sum(sizes),
            'avg_packet_size': np.mean(sizes),
            'std_packet_size': np.std(sizes),
            'min_packet_size': min(sizes),
            'max_packet_size': max(sizes),
            'median_packet_size': np.median(sizes),
            'session_duration': timestamps[-1] - timestamps[0],
            'avg_interval': np.mean(intervals) if len(intervals) > 0 else 0,
            'std_interval': np.std(intervals) if len(intervals) > 0 else 0,
            'min_interval': min(intervals) if len(intervals) > 0 else 0,
            'max_interval': max(intervals) if len(intervals) > 0 else 0,
            'tcp_packets': len(packet_data),  # Assuming all TCP
            'outgoing_ratio': 0.5,  # Simplified
            'incoming_ratio': 0.5
        }
        
        # Size histogram
        hist, _ = np.histogram(sizes, bins=10, range=(0, 1500))
        for i, count in enumerate(hist):
            features[f'size_bin_{i}'] = count
        
        return features
    
    def _extract_website_type(self, url: str) -> str:
        clean_url = url.replace('https://', '').replace('http://', '').replace('www.', '')
        domain = clean_url.split('/')[0].split('.')[0]
        return domain