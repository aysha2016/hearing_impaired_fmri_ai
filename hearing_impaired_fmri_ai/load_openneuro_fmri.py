import os
import numpy as np
import pandas as pd
from nilearn import datasets, input_data, image
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, Any
import nibabel as nib
import json

class FMRIProcessor:
    """Process and prepare fMRI data for integration with RL agent."""
    
    def __init__(self, 
                 data_dir: str = './data/fmri',
                 smoothing_fwhm: float = 5.0,
                 standardize: bool = True):
        self.data_dir = data_dir
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.masker = None
        self.scaler = StandardScaler() if standardize else None
        
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_data(self, dataset_name: str = 'development_fmri', 
                  n_subjects: int = 1) -> Dict[str, Any]:
        """Fetch fMRI data from OpenNeuro."""
        try:
            if dataset_name == 'development_fmri':
                dataset = datasets.fetch_development_fmri(n_subjects=n_subjects)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
            return {
                'func': dataset.func,
                'phenotypic': dataset.phenotypic,
                'description': dataset.description
            }
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise
    
    def preprocess_data(self, fmri_filename: str) -> np.ndarray:
        """Preprocess fMRI data."""
        # Initialize masker
        self.masker = input_data.NiftiMasker(
            smoothing_fwhm=self.smoothing_fwhm,
            standardize=self.standardize,
            detrend=True,
            high_pass=0.01,
            low_pass=0.1,
            t_r=2.0  # repetition time in seconds
        )
        
        # Load and preprocess data
        fmri_signals = self.masker.fit_transform(fmri_filename)
        
        if self.scaler is not None:
            fmri_signals = self.scaler.fit_transform(fmri_signals)
            
        return fmri_signals
    
    def extract_roi_signals(self, 
                          fmri_signals: np.ndarray,
                          roi_mask: str = None) -> np.ndarray:
        """Extract signals from regions of interest."""
        if roi_mask is None:
            # Use default auditory cortex mask
            roi_mask = datasets.fetch_neurovault_motor_task()['images'][0]
            
        roi_masker = input_data.NiftiMasker(
            mask_img=roi_mask,
            smoothing_fwhm=self.smoothing_fwhm,
            standardize=self.standardize
        )
        
        return roi_masker.fit_transform(fmri_signals)
    
    def compute_functional_connectivity(self, 
                                     fmri_signals: np.ndarray,
                                     kind: str = 'correlation') -> np.ndarray:
        """Compute functional connectivity between regions."""
        connectivity = ConnectivityMeasure(kind=kind)
        return connectivity.fit_transform([fmri_signals])[0]
    
    def save_processed_data(self, 
                          data: np.ndarray,
                          filename: str,
                          metadata: Dict[str, Any] = None):
        """Save processed data and metadata."""
        # Save data
        np.save(os.path.join(self.data_dir, filename), data)
        
        # Save metadata if provided
        if metadata is not None:
            meta_filename = os.path.join(self.data_dir, f"{filename}_meta.json")
            with open(meta_filename, 'w') as f:
                json.dump(metadata, f, indent=2)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize processor
    processor = FMRIProcessor()
    
    try:
        # Fetch data
        logger.info("Fetching fMRI data...")
        dataset = processor.fetch_data(n_subjects=1)
        
        # Process each subject's data
        for i, fmri_file in enumerate(dataset['func']):
            logger.info(f"Processing subject {i+1}...")
            
            # Preprocess
            fmri_signals = processor.preprocess_data(fmri_file)
            
            # Extract ROI signals
            roi_signals = processor.extract_roi_signals(fmri_signals)
            
            # Compute connectivity
            connectivity = processor.compute_functional_connectivity(roi_signals)
            
            # Save processed data
            metadata = {
                'subject_id': i,
                'preprocessing': {
                    'smoothing_fwhm': processor.smoothing_fwhm,
                    'standardize': processor.standardize
                }
            }
            
            processor.save_processed_data(
                roi_signals,
                f'subject_{i}_roi_signals.npy',
                metadata
            )
            
            processor.save_processed_data(
                connectivity,
                f'subject_{i}_connectivity.npy',
                metadata
            )
            
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()