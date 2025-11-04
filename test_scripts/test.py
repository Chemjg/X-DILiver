# -*- coding: utf-8 -*-
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from pandas import read_csv
import pickle
import tensorflow as tf
import joblib
import argparse
from collections import OrderedDict

import sys
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from basic_utils import normalizeData

# Logging settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ==================== Configuration ====================
@dataclass
class ModelConfig:
    """Individual model configuration"""
    name: str
    model_type: str  # 'xgb', 'lstm', 'gru'
    model_path: str
    dataset: str  # 'NF', 'HC', 'TS'
    
    # For XGBoost model (descriptor-based)
    fset_path: Optional[str] = None
    avg_path: Optional[str] = None
    std_path: Optional[str] = None
    
    # For LSTM/GRU model (structure-based)
    embedding_file: Optional[str] = None


class DatasetType(Enum):
    """Dataset type"""
    NF = "NF"  # No Filtering
    HC = "HC"  # Hierarchical Clustering
    TS = "TS"  # Tanimoto Similarity


class ModelType(Enum):
    """Model architecture type"""
    XGB = "xgb"
    LSTM = "lstm"
    GRU = "gru"


# ==================== Temporary File Manager ====================
class TemporaryFileManager:
    """Temporary file management class"""
    
    # Temporary files to clean up
    EMBEDDING_PATTERNS = [
        'DILI_NF_Maxsmi_N_gram_3_External.txt',
        'DILI_NF_N_gram_3_External.txt',
        'DILI_HC_Maxsmi_N_gram_3_External.txt',
        'DILI_HC_N_gram_2_External.txt',
        'DILI_TS_Maxsmi_N_gram_2_External.txt',
        'DILI_TS_N_gram_2_External.txt',
    ]
    
    @classmethod
    def cleanup(cls, work_dir: str) -> None:
        """Clean up temporary embedding files"""
        work_path = Path(work_dir)
        
        logger.info("Cleaning up temporary files...")
        
        # Delete embedding files
        for pattern in cls.EMBEDDING_PATTERNS:
            file_path = work_path / pattern
            if file_path.exists():
                try:
                    os.remove(str(file_path))
                    logger.info(f"Deleted file: {pattern}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {pattern}: {e}")
        
        logger.info("Temporary file cleanup completed")


# ==================== Path Manager ====================
class PathManager:
    """Path management class"""
    
    def __init__(self, data_dir: str = './Dat', work_dir: str = './'):
        self.data_dir = Path(data_dir)
        self.work_dir = Path(work_dir)

    def get_feature_set_path(self, fset_name: str) -> Path:
        """Feature set file path"""
        return self.data_dir / 'fset' / f'{fset_name}_fset.txt'
    
    def get_vectorizer_path(self, vectorizer_name: str) -> Path:
        """Vectorizer path"""
        return self.data_dir / 'Vectorizer' / f'{vectorizer_name}.pickle'
    
    def get_embedding_file(self, embedding_name: str) -> Path:
        """Embedding file path"""
        return self.work_dir / f'{embedding_name}_External.txt'
    
    def get_output_file(self, output_filename: str) -> Path:
        """Output file path"""
        return self.work_dir / output_filename


# ==================== Model Registry ====================
class ModelRegistry:
    """Model registry - centrally manage all model configurations"""
    
    def __init__(self, data_dir: str = './Dat'):
        self.data_dir = Path(data_dir)
        self._registry = self._build_registry()
    
    def _build_registry(self) -> Dict[str, ModelConfig]:
        """Build model registry"""
        registry = {}
        
        # ===== NF (No Finding) Models =====
        registry['NF_XGB'] = ModelConfig(
            name='NF_XGB',
            model_type=ModelType.XGB.value,
            model_path=str(self.data_dir / 'NF/Raw/DILI_NF_XGB.pkl'),
            dataset=DatasetType.NF.value,
            fset_path=str(self.data_dir / 'NF/Raw/data/DILI_NF_FS_fset.txt'),
            avg_path=str(self.data_dir / 'NF/Raw/data/avgFile.txt'),
            std_path=str(self.data_dir / 'NF/Raw/data/stdFile.txt')
        )
        
        registry['NF_KNNOR_LSTM_n=3'] = ModelConfig(
            name='NF_KNNOR_LSTM_n=3',
            model_type=ModelType.LSTM.value,
            model_path=str(self.data_dir / 'NF/KNNOR/DILI_NF_KNNOR_LSTM_n=3.h5'),
            dataset=DatasetType.NF.value,
            embedding_file='DILI_NF_N_gram_3'
        )
        
        registry['NF_Maxsmi_GRU_n=3'] = ModelConfig(
            name='NF_Maxsmi_GRU_n=3',
            model_type=ModelType.GRU.value,
            model_path=str(self.data_dir / 'NF/Maxsmi/DILI_NF_Maxsmi_GRU_n=3.h5'),
            dataset=DatasetType.NF.value,
            embedding_file='DILI_NF_Maxsmi_N_gram_3'
        )
        
        # ===== HC (Hepatotoxic Compound) Models =====
        registry['HC_LSTM_n=2'] = ModelConfig(
            name='HC_LSTM_n=2',
            model_type=ModelType.LSTM.value,
            model_path=str(self.data_dir / 'HC/Raw/DILI_HC_LSTM_n=2.h5'),
            dataset=DatasetType.HC.value,
            embedding_file='DILI_HC_N_gram_2'
        )
        
        registry['HC_KNNOR_LSTM_n=2'] = ModelConfig(
            name='HC_KNNOR_LSTM_n=2',
            model_type=ModelType.LSTM.value,
            model_path=str(self.data_dir / 'HC/KNNOR/DILI_HC_KNNOR_LSTM_n=2.h5'),
            dataset=DatasetType.HC.value,
            embedding_file='DILI_HC_N_gram_2'
        )
        
        registry['HC_Maxsmi_GRU_n=3'] = ModelConfig(
            name='HC_Maxsmi_GRU_n=3',
            model_type=ModelType.GRU.value,
            model_path=str(self.data_dir / 'HC/Maxsmi/DILI_HC_Maxsmi_GRU_n=3.h5'),
            dataset=DatasetType.HC.value,
            embedding_file='DILI_HC_Maxsmi_N_gram_3'
        )
        
        # ===== TS (Therapeutic Surrogate) Models =====
        registry['TS_XGB'] = ModelConfig(
            name='TS_XGB',
            model_type=ModelType.XGB.value,
            model_path=str(self.data_dir / 'TS/Raw/DILI_TS_XGB.pkl'),
            dataset=DatasetType.TS.value,
            fset_path=str(self.data_dir / 'TS/Raw/data/DILI_TS_FS_fset.txt'),
            avg_path=str(self.data_dir / 'TS/Raw/data/avgFile.txt'),
            std_path=str(self.data_dir / 'TS/Raw/data/stdFile.txt')
        )
        
        registry['TS_LSTM_n=2'] = ModelConfig(
            name='TS_LSTM_n=2',
            model_type=ModelType.LSTM.value,
            model_path=str(self.data_dir / 'TS/Raw/DILI_TS_LSTM_n=2.h5'),
            dataset=DatasetType.TS.value,
            embedding_file='DILI_TS_N_gram_2'
        )
        
        registry['TS_Maxsmi_GRU_n=2'] = ModelConfig(
            name='TS_Maxsmi_GRU_n=2',
            model_type=ModelType.GRU.value,
            model_path=str(self.data_dir / 'TS/Maxsmi/DILI_TS_Maxsmi_GRU_n=2.h5'),
            dataset=DatasetType.TS.value,
            embedding_file='DILI_TS_Maxsmi_N_gram_2'
        )
        
        return registry
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration"""
        if model_name not in self._registry:
            raise ValueError(f"Unknown model: {model_name}")
        return self._registry[model_name]
    
    def get_all_models(self) -> List[str]:
        """Return all model names"""
        return list(self._registry.keys())


# ==================== Embedding Generator ====================
class EmbeddingGenerator:
    """TF-IDF word embedding generation from SELFIES"""
    
    VECTORIZER_LIST = [
        'DILI_NF_Maxsmi_N_gram_3',
        'DILI_NF_N_gram_3',
        'DILI_HC_Maxsmi_N_gram_3',
        'DILI_HC_N_gram_2',
        'DILI_TS_Maxsmi_N_gram_2',
        'DILI_TS_N_gram_2'
    ]
    
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
    
    def generate_embeddings(self, smiles_list: List[str], work_dir: str) -> None:
        """Generate embeddings for all vectorizers from SMILES list"""
        # Convert SMILES to SELFIES on-the-fly
        import selfies as sf
        selfies_list = [sf.encoder(smi) for smi in smiles_list]
        
        for vec_name in self.VECTORIZER_LIST:
            self._generate_single_embedding(vec_name, selfies_list, work_dir)
        
        logger.info(f"Generated {len(self.VECTORIZER_LIST)} embeddings")
    
    def _generate_single_embedding(self, vec_name: str, selfies_list: List[str], 
                                   work_dir: str) -> None:
        """Generate embedding for single vectorizer"""
        vectorizer_path = self.path_manager.get_vectorizer_path(vec_name)
        
        with open(str(vectorizer_path), "rb") as f:
            vectorizer = pickle.load(f)
        
        tfidf_matrix = vectorizer.transform(selfies_list)
        tfidf_labels = list(range(1, tfidf_matrix.shape[1] + 1))
        
        output_path = Path(work_dir) / f'{vec_name}_External.txt'
        with open(str(output_path), 'w') as f:
            f.write('SELFIES\t' + "TF-IDF_" + 
                   '\tTF-IDF_'.join(map(str, tfidf_labels)) + '\n')
            
            for i in range(len(selfies_list)):
                tfidf = tfidf_matrix.getrow(i).toarray()[0]
                values_str = '\t'.join([str(value) for value in tfidf])
                f.write(f"{selfies_list[i]}\t{values_str}\n")


# ==================== Model Predictor ====================
class ModelPredictor:

    @staticmethod
    def predict_xgb(test_file: str, config: ModelConfig) -> np.ndarray:
        """Make predictions with XGBoost model (descriptor-based)"""
        X_test = read_csv(test_file, delimiter='\t',
                         usecols=lambda column: column != 'NAME', 
                         dtype='float32', skiprows=0, comment=None).values

        Nor_X_test = normalizeData(X_test, config.avg_path, config.std_path)
        
        # Reconstruct data for feature selection
        with open(test_file, 'r') as file:
            lines = file.readlines()
        first_row = lines[0].strip().split('\t')
        first_column = [line.split('\t')[0] for line in lines]
        combined_array = np.column_stack((first_column[1:], Nor_X_test))
        combined_array = np.vstack((first_row, combined_array))
        
        test_df = pd.DataFrame(combined_array)
        new_columns = test_df.iloc[0]
        test_df = test_df[1:]
        test_df.columns = new_columns
        
        # Feature selection
        with open(config.fset_path, 'r') as file:
            columns_list = file.read().strip().split('\t')
        
        num_features = len(columns_list)
        X_test = test_df[columns_list].iloc[:, :num_features].values
        
        # Load model and predict
        with open(config.model_path, 'rb') as h:
            model = joblib.load(h)
        
        predictions = model.predict_proba(X_test)[:, 1]
        return predictions.reshape(-1, 1)
    
    @staticmethod
    def predict_lstm_gru(embedding_file: str, config: ModelConfig) -> np.ndarray:
        """Make predictions with LSTM/GRU model (structure-based)"""
        X_test = read_csv(embedding_file, delimiter='\t',
                         usecols=lambda column: column != 'SELFIES', 
                         dtype='float32', skiprows=0, comment=None).values
        
        model = tf.keras.models.load_model(config.model_path)
        
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        predictions = model.predict(X_test, verbose=0)
        return predictions


# ==================== Ensemble Voter ====================
class EnsembleVoter:
    """Ensemble voting class"""
    
    def __init__(self, model_registry: ModelRegistry, path_manager: PathManager):
        self.model_registry = model_registry
        self.path_manager = path_manager
        self.predictor = ModelPredictor()
    
    def vote_soft(self, test_file: str, smiles_list: List[str], 
                  work_dir: str) -> Dict[str, str]:
        """Perform soft voting (average probability)"""
        predictions = []
        
        # Get predictions from all models
        for model_name in self.model_registry.get_all_models():
            config = self.model_registry.get_model_config(model_name)
            try:
                pred = self._get_prediction(config, test_file, work_dir)
                predictions.append(pred)
                logger.info(f"Model {model_name} prediction completed")
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        if len(predictions) == 0:
            raise ValueError("No models could be loaded for prediction")
        
        # Average predictions
        predictions = np.array(predictions)
        average_probs = np.mean(predictions, axis=0)
        flattened_probs = np.array(average_probs).flatten().tolist()
        prob_list = [f"{round(num, 3):.3f}" for num in flattened_probs]
        
        # Map SMILES to prediction results
        result = {smiles_list[i]: prob_list[i] for i in range(len(prob_list))}
        
        logger.info(f"Ensemble voting completed for {len(result)} molecules using {len(predictions)} models")
        return result
    
    def _get_prediction(self, config: ModelConfig, test_file: str, 
                       work_dir: str) -> np.ndarray:
        """Get predictions based on model type"""
        if config.model_type == ModelType.XGB.value:
            return self.predictor.predict_xgb(test_file, config)
        else:
            embedding_file = str(self.path_manager.get_embedding_file(
                config.embedding_file
            ))
            return self.predictor.predict_lstm_gru(embedding_file, config)


# ==================== Result Writer ====================
class ResultWriter:
    """Result writing class - supports Tab-separated format (TSV)"""
    
    @staticmethod
    def write_results(output_file: str, results: Dict[str, str]) -> None:
        """
        Write results to output file in Tab-separated format (TSV)
        
        Args:
            output_file: Path to output file
            results: Dictionary mapping SMILES to toxicity probability
        
        Format: SMILES\tProbability\tPrediction
        """
        try:
            with open(output_file, 'w') as f:
                # Write header
                f.write("SMILES\tProbability\tPrediction\n")
                
                # Write results
                for smiles, probability in results.items():
                    # Convert probability to float and classify
                    prob_value = float(probability)
                    
                    # Classify based on probability threshold
                    if prob_value < 0.25:
                        prediction = "Highly Safe"
                    elif prob_value < 0.50:
                        prediction = "Safe"
                    elif prob_value < 0.75:
                        prediction = "Less DILI concern"
                    else:
                        prediction = "Highly Toxic"
                    
                    f.write(f"{smiles}\t{probability}\t{prediction}\n")
            
            logger.info(f"Results saved to {output_file}")
            logger.info(f"Total predictions: {len(results)}")
            
        except IOError as e:
            logger.error(f"Failed to write results to {output_file}: {e}")
            raise


# ==================== Descriptor-based Predictor ====================
class DescriptorPredictor:
    """DILI prediction with pre-calculated descriptors"""
    
    def __init__(self, work_dir: str = './', data_dir: str = './Dat'):
        self.work_dir = work_dir
        self.path_manager = PathManager(data_dir, work_dir)
        self.model_registry = ModelRegistry(data_dir)
        self.embedding_generator = EmbeddingGenerator(self.path_manager)
        self.ensemble_voter = EnsembleVoter(self.model_registry, self.path_manager)
        self.result_writer = ResultWriter()

    def _get_smiles_from_descriptor_file(self, desc_file: str) -> List[str]:
        """Extract SMILES from descriptor file (NAME column)"""
        df = read_csv(desc_file, delimiter='\t', usecols=['NAME'])
        return df['NAME'].tolist()

    def predict(self, desc_filename: str, output_filename: str, cleanup: bool = True) -> OrderedDict:
        """
        Make predictions using pre-calculated descriptors
        
        Args:
            desc_filename: Input descriptor file path (e.g., DescFixed_rowChecked.txt)
            output_filename: Output file path for predictions
            cleanup: Whether to clean up temporary files (default: True)
        
        Returns:
            OrderedDict containing SMILES and toxicity predictions
        """
        
        desc_path = self.work_dir + '/' + desc_filename
        output_path = self.work_dir + '/' + output_filename
        
        try:
            logger.info(f"Starting descriptor-based prediction...")
            logger.info(f"Descriptor file: {desc_filename}")
            logger.info(f"Output file: {output_filename}")
            
            # Verify input file exists
            if not Path(desc_path).exists():
                logger.error(f"Descriptor file not found: {desc_filename}")
                raise FileNotFoundError(f"Descriptor file not found: {desc_filename}")
            
            # Create output directory if needed
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # ===== Step 1: Extract SMILES from descriptor file =====
            logger.info("Step 1: Extracting SMILES from descriptor file...")
            smiles_list = self._get_smiles_from_descriptor_file(desc_path)
            logger.info(f"Found {len(smiles_list)} molecules")
            
            # ===== Step 2: Generate embeddings from SMILES =====
            logger.info("Step 2: Generating embeddings from SMILES...")
            self.embedding_generator.generate_embeddings(smiles_list, self.work_dir)
            
            # ===== Step 3: Run Ensemble Prediction =====
            logger.info("Step 3: Running ensemble prediction...")
            result = self.ensemble_voter.vote_soft(desc_path, smiles_list, self.work_dir)
            
            # ===== Step 4: Preserve Order and Save Results =====
            logger.info("Step 4: Saving results...")
            
            # Preserve order using OrderedDict
            ordered_result = OrderedDict()
            for smiles in smiles_list:
                if smiles in result:
                    ordered_result[smiles] = result[smiles]
            
            # Write results to file
            self.result_writer.write_results(output_path, ordered_result)
            
            logger.info(f"Prediction completed for {len(ordered_result)} molecules")
            logger.info(f"Results saved to: {output_path}")
            
            # ===== Step 5: Clean up temporary files =====
            if cleanup:
                logger.info("Step 5: Cleaning up temporary files...")
                TemporaryFileManager.cleanup(self.work_dir)
            
            return ordered_result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='X-DILiver: Descriptor-based DILI Prediction (Full Ensemble)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py -i DescFixed_rowChecked.txt -o result.txt
  python test.py --input descriptor_file.txt --output output.txt --work-dir ./ --data-dir ./Dat -v
        """
    )

    parser.add_argument(
        '-i', '--input',
        dest='input_file',
        required=True,
        help='Input descriptor file path (e.g., DescFixed_rowChecked.txt)'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        required=True,
        help='Output file path for predictions'
    )

    parser.add_argument(
        '--work-dir',
        dest='work_dir',
        default='./',
        help='Working directory path (default: ./)'
    )

    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        default='./Dat',
        help='Data directory path (default: ./Dat)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()
    return args


# ==================== Main Execution ====================
def main():
    """Main execution function"""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info(f"Starting X-DILiver descriptor-based prediction...")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")

    # Initialize predictor
    predictor = DescriptorPredictor(work_dir=args.work_dir, data_dir=args.data_dir)

    # Run prediction
    try:
        result = predictor.predict(
            desc_filename=args.input_file,
            output_filename=args.output_file,
            cleanup=True
        )
        logger.info(f"Prediction completed successfully!")
        logger.info(f"Total molecules predicted: {len(result)}")
        logger.info(f"Results saved to: {args.output_file}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
