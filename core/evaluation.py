import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up plotting preferences for consistent, professional-looking charts
        plt.style.use('default')
        sns.set_palette("husl")
    
    def analyze_results(self, model_results: Dict[str, Any], 
                       features_df: pd.DataFrame, labels: List[str]) -> Dict[str, Any]:
        logger.info("Analyzing model results...")
        
        analysis = {
            'model_comparison': self._compare_models(model_results),
            'feature_analysis': self._analyze_features(model_results, features_df),
            'confusion_matrices': self._generate_confusion_matrices(model_results),
            'classification_reports': self._generate_classification_reports(model_results)
        }
        
        # Create plots and charts to visualize the analysis results
        self._plot_model_comparison(analysis['model_comparison'])
        self._plot_feature_importance(analysis['feature_analysis'])
        self._plot_confusion_matrices(analysis['confusion_matrices'])
        
        return analysis
    
    def _compare_models(self, model_results: Dict[str, Any]) -> pd.DataFrame:
        comparison_data = []
        
        for model_name, model_info in model_results.items():
            if 'accuracy' in model_info:
                # Calculate additional metrics if predictions available
                metrics = {'Model': model_name, 'Accuracy': model_info['accuracy']}
                
                # Calculate detailed metrics if test predictions are available
                if 'predictions' in model_info and 'y_test' in model_info:
                    y_true = model_info['y_test']
                    y_pred = model_info['predictions']
                    
                    # Compute comprehensive classification metrics
                    report = classification_report(y_true, y_pred, output_dict=True)
                    metrics.update({
                        'Precision': report['macro avg']['precision'],
                        'Recall': report['macro avg']['recall'],
                        'F1-Score': report['macro avg']['f1-score']
                    })
                
                comparison_data.append(metrics)
        
        return pd.DataFrame(comparison_data)
    
    def _analyze_features(self, model_results: Dict[str, Any], 
                         features_df: pd.DataFrame) -> Dict[str, Any]:
        feature_analysis = {}
        feature_names = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for model_name, model_info in model_results.items():
            model = model_info.get('model')
            
            # Extract feature importance if the model supports it (tree-based models)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Account for any feature selection that was applied during training
                if 'feature_selector' in model_info:
                    selector = model_info['feature_selector']
                    selected_indices = selector.get_support(indices=True)
                    selected_names = [feature_names[i] for i in selected_indices]
                else:
                    selected_names = feature_names[:len(importances)]
                
                importance_dict = dict(zip(selected_names, importances))
                feature_analysis[model_name] = dict(sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                ))
        
        return feature_analysis
    
    def _generate_confusion_matrices(self, model_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        matrices = {}
        
        for model_name, model_info in model_results.items():
            if 'predictions' in model_info and 'y_test' in model_info:
                y_true = model_info['y_test']
                y_pred = model_info['predictions']
                
                conf_matrix = confusion_matrix(y_true, y_pred)
                matrices[model_name] = conf_matrix
        
        return matrices
    
    def _generate_classification_reports(self, model_results: Dict[str, Any]) -> Dict[str, Dict]:
        reports = {}
        
        for model_name, model_info in model_results.items():
            if 'predictions' in model_info and 'y_test' in model_info:
                y_true = model_info['y_test']
                y_pred = model_info['predictions']
                
                # Get class names
                if 'label_encoder' in model_info:
                    class_names = model_info['label_encoder'].classes_
                else:
                    class_names = None
                
                report = classification_report(y_true, y_pred, 
                                             target_names=class_names,
                                             output_dict=True)
                reports[model_name] = report
        
        return reports
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        if comparison_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i // 2, i % 2]
            
            if metric in comparison_df.columns:
                bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=color, alpha=0.7)
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, comparison_df[metric]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, f'{metric}\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric} Comparison')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison plot saved")
    
    def _plot_feature_importance(self, feature_analysis: Dict[str, Any]):
        if not feature_analysis:
            return
        
        # Plot for each model that has feature importance
        for model_name, importances in feature_analysis.items():
            if not importances:
                continue
            
            # Get top 20 features
            top_features = dict(list(importances.items())[:20])
            
            plt.figure(figsize=(12, 8))
            features = list(top_features.keys())
            values = list(top_features.values())
            
            bars = plt.barh(range(len(features)), values, color='steelblue', alpha=0.7)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title(f'Top 20 Feature Importances - {model_name}')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', va='center')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'feature_importance_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Feature importance plots saved")
    
    def _plot_confusion_matrices(self, matrices: Dict[str, np.ndarray]):
        if not matrices:
            return
        
        for model_name, matrix in matrices.items():
            plt.figure(figsize=(8, 6))
            
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                       cbar_kws={'label': 'Count'})
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'confusion_matrix_{model_name}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Confusion matrix plots saved")
    
    def generate_summary_report(self, analysis: Dict[str, Any], 
                               dataset_info: Dict[str, Any]) -> str:
        report_lines = [
            "BABBAGE TRAFFIC FINGERPRINTING - ANALYSIS REPORT",
            "=" * 60,
            "",
            "DATASET INFORMATION:",
            f"  Total Samples: {dataset_info.get('total_samples', 'N/A')}",
            f"  Number of Classes: {dataset_info.get('num_classes', 'N/A')}",
            f"  Features: {dataset_info.get('num_features', 'N/A')}",
            "",
            "MODEL PERFORMANCE COMPARISON:",
        ]
        
        # Add model comparison
        if 'model_comparison' in analysis:
            comparison_df = analysis['model_comparison']
            for _, row in comparison_df.iterrows():
                model_name = row['Model']
                accuracy = row.get('Accuracy', 'N/A')
                report_lines.append(f"  {model_name:20s}: {accuracy:.4f}")
        
        report_lines.extend([
            "",
            "TOP FEATURES (Overall Importance):",
        ])
        
        # Aggregate feature importance
        if 'feature_analysis' in analysis:
            all_features = {}
            for model_features in analysis['feature_analysis'].values():
                for feature, importance in model_features.items():
                    if feature in all_features:
                        all_features[feature].append(importance)
                    else:
                        all_features[feature] = [importance]
            
            # Average importance across models
            avg_importance = {f: np.mean(imp) for f, imp in all_features.items()}
            top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for i, (feature, importance) in enumerate(top_features, 1):
                report_lines.append(f"  {i:2d}. {feature:25s}: {importance:.4f}")
        
        report_lines.extend([
            "",
            "CLASSIFICATION PERFORMANCE:",
        ])
        
        # Add detailed classification metrics
        if 'classification_reports' in analysis:
            for model_name, report in analysis['classification_reports'].items():
                if 'macro avg' in report:
                    macro_avg = report['macro avg']
                    report_lines.extend([
                        f"  {model_name}:",
                        f"    Precision: {macro_avg['precision']:.4f}",
                        f"    Recall:    {macro_avg['recall']:.4f}",
                        f"    F1-Score:  {macro_avg['f1-score']:.4f}",
                        ""
                    ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_dir / 'analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Summary report saved to {report_file}")
        return report_content
    
    def save_results_json(self, analysis: Dict[str, Any], filename: str = 'results.json'):
        # Convert numpy arrays to lists for JSON serialization
        json_safe_analysis = self._make_json_safe(analysis)
        
        results_file = self.results_dir / filename
        with open(results_file, 'w') as f:
            json.dump(json_safe_analysis, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def _make_json_safe(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def plot_training_history(self, history: Dict[str, List[float]], model_name: str):
        if not history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        if 'accuracy' in history and 'val_accuracy' in history:
            axes[0].plot(history['accuracy'], label='Training Accuracy')
            axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
            axes[0].set_title(f'{model_name} - Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
        
        # Plot loss
        if 'loss' in history and 'val_loss' in history:
            axes[1].plot(history['loss'], label='Training Loss')
            axes[1].plot(history['val_loss'], label='Validation Loss')
            axes[1].set_title(f'{model_name} - Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'training_history_{model_name}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved for {model_name}")
    
    def compare_data_sources(self, results_by_source: Dict[str, Dict[str, Any]]):
        comparison_data = []
        
        for source, model_results in results_by_source.items():
            for model_name, model_info in model_results.items():
                if 'accuracy' in model_info:
                    comparison_data.append({
                        'Data Source': source,
                        'Model': model_name,
                        'Accuracy': model_info['accuracy']
                    })
        
        if not comparison_data:
            return
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar chart
        plt.figure(figsize=(12, 8))
        
        # Pivot for easier plotting
        pivot_df = comparison_df.pivot(index='Model', columns='Data Source', values='Accuracy')
        
        ax = pivot_df.plot(kind='bar', width=0.8, alpha=0.7)
        plt.title('Model Performance Across Data Sources')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.legend(title='Data Source')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.results_dir / 'data_source_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Data source comparison plot saved")