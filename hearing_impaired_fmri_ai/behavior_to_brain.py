import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, Any
import json
import os

class BehaviorBrainMapper(nn.Module):
    """Neural network to map agent behavior to brain activity patterns."""
    
    def __init__(self, 
                 behavior_dim: int,
                 brain_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 128)):
        super().__init__()
        
        # Build network layers
        layers = []
        prev_dim = behavior_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, brain_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class BehaviorBrainMapping:
    """Class to handle mapping between agent behavior and brain activity."""
    
    def __init__(self, 
                 model_dir: str = './models/behavior_brain',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.scaler = StandardScaler()
        
        os.makedirs(model_dir, exist_ok=True)
        
    def prepare_data(self, 
                    behavior_data: np.ndarray,
                    brain_data: np.ndarray) -> Tuple[torch.Tensor, ...]:
        """Prepare data for training."""
        # Scale data
        behavior_scaled = self.scaler.fit_transform(behavior_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            behavior_scaled, brain_data, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        return X_train, X_test, y_train, y_test
    
    def train(self,
             behavior_data: np.ndarray,
             brain_data: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32,
             learning_rate: float = 0.001):
        """Train the mapping model."""
        # Initialize model
        self.model = BehaviorBrainMapper(
            behavior_dim=behavior_data.shape[1],
            brain_dim=brain_data.shape[1]
        ).to(self.device)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            behavior_data, brain_data
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_test_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = criterion(test_outputs, y_test)
                
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    self.save_model()
            
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}, "
                           f"Train Loss: {total_loss/len(X_train):.4f}, "
                           f"Test Loss: {test_loss:.4f}")
    
    def predict(self, behavior_data: np.ndarray) -> np.ndarray:
        """Predict brain activity from behavior."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Scale input
        behavior_scaled = self.scaler.transform(behavior_data)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                torch.FloatTensor(behavior_scaled).to(self.device)
            )
            
        return predictions.cpu().numpy()
    
    def save_model(self):
        """Save model and scaler."""
        if self.model is None:
            raise ValueError("No model to save!")
            
        # Save model
        torch.save(self.model.state_dict(),
                  os.path.join(self.model_dir, 'model.pt'))
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler,
                   os.path.join(self.model_dir, 'scaler.joblib'))
        
    def load_model(self):
        """Load saved model and scaler."""
        # Load model
        self.model = BehaviorBrainMapper(
            behavior_dim=self.scaler.n_features_in_,
            brain_dim=0  # Will be set when loading weights
        )
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'model.pt'))
        )
        self.model.to(self.device)
        
        # Load scaler
        import joblib
        self.scaler = joblib.load(
            os.path.join(self.model_dir, 'scaler.joblib')
        )

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Example usage
    try:
        # Load your data here
        behavior_data = np.load('agent_behavior.npy')
        brain_data = np.load('fmri_signals.npy')
        
        # Initialize mapper
        mapper = BehaviorBrainMapping()
        
        # Train model
        logger.info("Training behavior-brain mapping model...")
        mapper.train(behavior_data, brain_data)
        
        # Make predictions
        predictions = mapper.predict(behavior_data)
        logger.info(f"Predictions shape: {predictions.shape}")
        
    except Exception as e:
        logger.error(f"Error in mapping: {str(e)}")
        raise

if __name__ == "__main__":
    main()