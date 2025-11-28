"""
VPN vs TOR Network Traffic Classifier
Main training and evaluation script
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime


class VPNTORClassifier:
    """
    A machine learning classifier for distinguishing between VPN and TOR traffic.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the classifier.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'gradient_boosting', 'svm')
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._initialize_model()
        self.feature_names = None
        self.history = {}
        
    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic network traffic data for demonstration.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        X : DataFrame
            Feature matrix
        y : array
            Labels (0=VPN, 1=TOR)
        """
        np.random.seed(42)
        
        # Generate VPN traffic characteristics (label=0)
        vpn_samples = n_samples // 2
        vpn_data = {
            'packet_size_mean': np.random.normal(800, 200, vpn_samples),
            'packet_size_std': np.random.normal(300, 50, vpn_samples),
            'flow_duration': np.random.exponential(15, vpn_samples),
            'packets_per_second': np.random.normal(50, 15, vpn_samples),
            'bytes_per_second': np.random.normal(40000, 10000, vpn_samples),
            'packet_iat_mean': np.random.exponential(0.02, vpn_samples),
            'packet_iat_std': np.random.exponential(0.05, vpn_samples),
            'forward_packets': np.random.poisson(100, vpn_samples),
            'backward_packets': np.random.poisson(90, vpn_samples),
            'flow_bytes_per_sec': np.random.normal(45000, 12000, vpn_samples),
            'encryption_overhead': np.random.normal(1.15, 0.05, vpn_samples),
        }
        
        # Generate TOR traffic characteristics (label=1)
        tor_samples = n_samples - vpn_samples
        tor_data = {
            'packet_size_mean': np.random.normal(512, 100, tor_samples),  # Fixed cell size
            'packet_size_std': np.random.normal(50, 20, tor_samples),    # Lower variance
            'flow_duration': np.random.exponential(25, tor_samples),      # Longer sessions
            'packets_per_second': np.random.normal(30, 10, tor_samples),  # Lower rate
            'bytes_per_second': np.random.normal(15000, 5000, tor_samples),
            'packet_iat_mean': np.random.exponential(0.04, tor_samples),  # Higher latency
            'packet_iat_std': np.random.exponential(0.08, tor_samples),   # More variable
            'forward_packets': np.random.poisson(80, tor_samples),
            'backward_packets': np.random.poisson(75, tor_samples),
            'flow_bytes_per_sec': np.random.normal(18000, 6000, tor_samples),
            'encryption_overhead': np.random.normal(1.35, 0.08, tor_samples),  # Higher overhead
        }
        
        # Combine data
        vpn_df = pd.DataFrame(vpn_data)
        tor_df = pd.DataFrame(tor_data)
        
        X = pd.concat([vpn_df, tor_df], ignore_index=True)
        y = np.array([0] * vpn_samples + [1] * tor_samples)
        
        # Add some noise and edge cases for realism
        X = X + np.random.normal(0, X.std() * 0.05, X.shape)
        X = X.clip(lower=0)  # No negative values
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the classifier.
        
        Parameters:
        -----------
        X_train : DataFrame or array
            Training features
        y_train : array
            Training labels
        X_val : DataFrame or array, optional
            Validation features
        y_val : array, optional
            Validation labels
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} classifier...")
        print(f"{'='*60}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Training metrics
        train_score = self.model.score(X_train_scaled, y_train)
        print(f"\nTraining Accuracy: {train_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
            print(f"Validation Accuracy: {val_score:.4f}")
        
        self.history = {
            'train_score': train_score,
            'cv_scores': cv_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}\n")
    
    def evaluate(self, X_test, y_test, output_dir='results'):
        """
        Evaluate the classifier and generate visualizations.
        
        Parameters:
        -----------
        X_test : DataFrame or array
            Test features
        y_test : array
            Test labels
        output_dir : str
            Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        report = classification_report(y_test, y_pred, 
                                       target_names=['VPN', 'TOR'],
                                       digits=4)
        print(report)
        
        # Save report
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Confusion Matrix
        self._plot_confusion_matrix(y_test, y_pred, output_dir)
        
        # ROC Curve
        self._plot_roc_curve(y_test, y_pred_proba, output_dir)
        
        # Feature Importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance(output_dir)
        
        # Prediction distribution
        self._plot_prediction_distribution(y_test, y_pred_proba, output_dir)
        
        return {
            'accuracy': np.mean(y_pred == y_test),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, output_dir):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['VPN', 'TOR'],
                   yticklabels=['VPN', 'TOR'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_type.upper()}', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved to {output_dir}/confusion_matrix.png")
    
    def _plot_roc_curve(self, y_true, y_pred_proba, output_dir):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_type.upper()}', fontsize=16, pad=20)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ ROC curve saved to {output_dir}/roc_curve.png")
    
    def _plot_feature_importance(self, output_dir):
        """Plot and save feature importance."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices], color='steelblue')
        plt.xticks(range(len(importances)), 
                  [self.feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.title(f'Feature Importance - {self.model_type.upper()}', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature importance saved to {output_dir}/feature_importance.png")
    
    def _plot_prediction_distribution(self, y_true, y_pred_proba, output_dir):
        """Plot prediction probability distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # VPN predictions
        vpn_mask = y_true == 0
        axes[0].hist(y_pred_proba[vpn_mask, 0], bins=50, alpha=0.7, 
                    color='blue', edgecolor='black', label='True VPN')
        axes[0].hist(y_pred_proba[~vpn_mask, 0], bins=50, alpha=0.7, 
                    color='red', edgecolor='black', label='True TOR')
        axes[0].set_xlabel('Predicted Probability (VPN)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('VPN Prediction Distribution', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # TOR predictions
        tor_mask = y_true == 1
        axes[1].hist(y_pred_proba[tor_mask, 1], bins=50, alpha=0.7, 
                    color='red', edgecolor='black', label='True TOR')
        axes[1].hist(y_pred_proba[~tor_mask, 1], bins=50, alpha=0.7, 
                    color='blue', edgecolor='black', label='True VPN')
        axes[1].set_xlabel('Predicted Probability (TOR)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('TOR Prediction Distribution', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Prediction distribution saved to {output_dir}/prediction_distribution.png")
    
    def save_model(self, filepath='model'):
        """Save the trained model and scaler."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'history': self.history
        }, f"{filepath}.pkl")
        print(f"\n✓ Model saved to {filepath}.pkl")
    
    @classmethod
    def load_model(cls, filepath='model.pkl'):
        """Load a trained model."""
        data = joblib.load(filepath)
        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.feature_names = data['feature_names']
        classifier.history = data['history']
        print(f"✓ Model loaded from {filepath}")
        return classifier
    
    def predict(self, X):
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'labels': ['VPN' if p == 0 else 'TOR' for p in predictions]
        }


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("VPN vs TOR Network Traffic Classifier")
    print("="*60)
    
    # Initialize classifier
    classifier = VPNTORClassifier(model_type='random_forest')
    
    # Generate synthetic data
    print("\n[1/5] Generating synthetic network traffic data...")
    X, y = classifier.generate_synthetic_data(n_samples=10000)
    print(f"✓ Generated {len(X)} samples with {X.shape[1]} features")
    print(f"  - VPN samples: {np.sum(y == 0)}")
    print(f"  - TOR samples: {np.sum(y == 1)}")
    
    # Split data
    print("\n[2/5] Splitting data into train/validation/test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    print(f"✓ Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model
    print("\n[3/5] Training model...")
    classifier.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("\n[4/5] Evaluating model...")
    results = classifier.evaluate(X_test, y_test, output_dir='results')
    print(f"\n✓ Test Accuracy: {results['accuracy']:.4f}")
    
    # Save model
    print("\n[5/5] Saving model...")
    classifier.save_model('models/vpn_tor_classifier')
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - models/vpn_tor_classifier.pkl (trained model)")
    print("  - results/confusion_matrix.png")
    print("  - results/roc_curve.png")
    print("  - results/feature_importance.png")
    print("  - results/prediction_distribution.png")
    print("  - results/classification_report.txt")
    print("\n")


if __name__ == "__main__":
    main()
