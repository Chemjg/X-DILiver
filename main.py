# -*- coding: utf-8 -*-
import pickle
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from pandas import read_csv
import tensorflow as tf
import selfies as sf
from rdkit import Chem, RDLogger
import joblib
import argparse
from collections import OrderedDict

import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from molecular_graph_utils import atom_feature
from molecular_desc_utils import (
    calculateMord, configGenerator, callDragon,
    featureFusionner, saveSelectedWithToxicity, NA_rowChecker
)
from basic_utils import (
    getList, getFeatures, normalizeData,
    MergeProcess
)

# RDKit and warning settings
RDLogger.DisableLog('rdApp.*')
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
class MolecularConfig:
    """Molecular structure configuration"""
    max_num_atoms: int = 155
    num_features: int = 102


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


class DILIPredictionClass(Enum):
    """DILI prediction classification based on probability"""
    HIGHLY_SAFE = "Highly Safe"
    SAFE = "Safe"
    LESS_DILI_CONCERN = "Less DILI concern"
    HIGHLY_TOXIC = "Highly Toxic"
    UNPREDICTABLE = "---"


# ==================== Path Manager ====================
class PathManager:
    """Path management class"""
    
    def __init__(self, data_dir: str = './Dat', work_dir: str = './'):
        self.data_dir = Path(data_dir)
        self.work_dir = Path(work_dir)
        self._custom_output_file = None

    def get_descriptor_test_file(self) -> Path:
        """Descriptor-based test file"""
        return self.work_dir / 'DescFixed_rowChecked.txt'
    
    def get_predictable_file(self) -> Path:
        """Predictable SMILES file"""
        return self.work_dir / 'Predictable_smi.txt'
    
    def get_predictable_checked_file(self) -> Path:
        """Predictable SMILES file after NA check"""
        return self.work_dir / 'Predictable_smi_checked.txt'
    
    def get_unpredictable_file(self) -> Path:
        """Unpredictable SMILES file"""
        return self.work_dir / 'UnPredictable_smi.txt'
    
    def get_selfies_file(self) -> Path:
        """SELFIES file"""
        return self.work_dir / 'SELFIES.txt'
    
    def get_embedding_file(self, embedding_name: str) -> Path:
        """Embedding file path"""
        return self.work_dir / f'{embedding_name}_External.txt'
    
    def get_vectorizer_path(self, vectorizer_name: str) -> Path:
        """Vectorizer path"""
        return self.data_dir / 'Vectorizer' / f'{vectorizer_name}.pickle'
    
    def get_feature_set_path(self, fset_name: str) -> Path:
        """Feature set file path"""
        return self.data_dir / 'fset' / f'{fset_name}_fset.txt'

    def get_output_file(self) -> Path:
        """Output file path"""
        if self._custom_output_file is not None:
            return self._custom_output_file
        return self.work_dir / 'output.txt'

    def set_output_file(self, output_path: str) -> None:
        """Set custom output file path"""
        self._custom_output_file = Path(output_path)

# ==================== Temporary File Manager ====================
class TemporaryFileManager:
    """Temporary file management class"""
    
    # Temporary directories
    TEMP_DIRECTORIES = [
        'Dragon_output',
        'dragon_output'
    ]
    
    # Temporary files
    TEMP_FILES = [
        'config.drs',
        'Log_Dra.txt',
        'Predictable_smi.txt',
        'UnPredictable_smi.txt',
        'SELFIES.txt',
        'Dsc_Mord.txt',
        'Dsc_Mord_fixed.txt',
        'Dsc_Dra.txt',
        'Dsc_Dra_fixed.txt',
        'Fusion_Dra_Mord.txt',
        'DescFixedReduced.txt',
        'DescFixed_rowChecked.txt',
        'Predictable_smi_checked.txt',
        'mol.sdf',
        'mol.mol2',
    ]
    
    # Embedding files (dynamic)
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
        """Clean up temporary files and directories"""
        work_path = Path(work_dir)
        
        logger.info("Cleaning up temporary files...")
        
        # 1. Delete temporary directories
        for dir_name in cls.TEMP_DIRECTORIES:
            dir_path = work_path / dir_name
            if dir_path.exists():
                try:
                    shutil.rmtree(str(dir_path))
                    logger.info(f"Deleted directory: {dir_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete directory {dir_name}: {e}")
        
        # 2. Delete temporary files
        for file_name in cls.TEMP_FILES:
            file_path = work_path / file_name
            if file_path.exists():
                try:
                    os.remove(str(file_path))
                    logger.info(f"Deleted file: {file_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_name}: {e}")
        
        # 3. Delete embedding files
        for pattern in cls.EMBEDDING_PATTERNS:
            file_path = work_path / pattern
            if file_path.exists():
                try:
                    os.remove(str(file_path))
                    logger.info(f"Deleted file: {pattern}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {pattern}: {e}")
        
        logger.info("Temporary file cleanup completed")
    
    @classmethod
    def get_files_to_cleanup(cls) -> List[str]:
        """Return list of files to clean up (for reference)"""
        all_files = cls.TEMP_FILES + cls.EMBEDDING_PATTERNS + cls.TEMP_DIRECTORIES
        return all_files


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
            fset_path=str(self.data_dir / 'NF/Raw/Dat/DILI_NF_FS_fset.txt'),
            avg_path=str(self.data_dir / 'NF/Raw/Dat/avgFile.txt'),
            std_path=str(self.data_dir / 'NF/Raw/Dat/stdFile.txt')
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
            fset_path=str(self.data_dir / 'TS/Raw/Dat/DILI_TS_FS_fset.txt'),
            avg_path=str(self.data_dir / 'TS/Raw/Dat/avgFile.txt'),
            std_path=str(self.data_dir / 'TS/Raw/Dat/stdFile.txt')
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


# ==================== SMILES Validator ====================
class SMILESValidator:
    """SMILES validation class"""
    
    def __init__(self, config: MolecularConfig):
        self.config = config
    
    @staticmethod
    def smiles_to_selfies(smiles: str) -> str:
        """Convert SMILES to SELFIES"""
        return sf.encoder(smiles)
    
    @staticmethod
    def is_valid_smiles(smiles: str) -> bool:
        """Check if SMILES can be converted to SELFIES"""
        try:
            sf.encoder(smiles)
            return True
        except:
            return False
    
    def validate_and_split(self, input_path: str, predictable_path: str, 
                          unpredictable_path: str) -> None:
        """Validate SMILES and split into predictable/unpredictable"""
        data_file = pd.read_csv(input_path, delimiter='\t', header=None, 
                               skiprows=0, comment=None)
        
        with open(predictable_path, 'w') as pred_f, \
             open(unpredictable_path, 'w') as unpred_f:
            
            for index, row in data_file.iterrows():
                smi = row[0].strip()
                
                # Step 1: Basic SMILES validity check
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    unpred_f.write(smi + "\n")
                    continue
                
                # Step 2: Atom count check
                num_atoms = mol.GetNumAtoms()
                if num_atoms > self.config.max_num_atoms:
                    unpred_f.write(smi + "\n")
                    continue
                
                # Step 3: SELFIES conversion validity check
                if self.is_valid_smiles(smi):
                    pred_f.write(smi + "\n")
                else:
                    unpred_f.write(smi + "\n")


# ==================== SELFIES Converter ====================
class SELFIESConverter:
    """SMILES to SELFIES conversion class"""
    
    @staticmethod
    def convert_file(input_path: str, output_path: str) -> None:
        """Convert SMILES in file to SELFIES"""
        smiles_list = getList(input_path)
        selfies_list = [sf.encoder(smi) for smi in smiles_list]
        
        with open(output_path, 'w') as f:
            f.write("SMILES\tSELFIES\n")
            for smi, selfies in zip(smiles_list, selfies_list):
                f.write(f"{smi}\t{selfies}\n")
        
        logger.info(f"Converted {len(smiles_list)} SMILES to SELFIES")


# ==================== Word Embedding Generator ====================
class WordEmbeddingGenerator:
    """TF-IDF word embedding generation class"""
    
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
    
    def generate_embeddings(self, selfies_file: str, work_dir: str) -> None:
        """Generate embeddings for all vectorizers"""
        data_file = pd.read_csv(selfies_file, delimiter='\t', skiprows=0, comment=None)
        selfies_list = data_file["SELFIES"].values.tolist()
        
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
    
    def vote_soft(self, test_file: str, predictable_file: str, 
                  work_dir: str) -> Dict[str, str]:
        """Perform soft voting (average probability)"""
        predictions = []
        
        # Get predictions from all models
        for model_name in self.model_registry.get_all_models():
            config = self.model_registry.get_model_config(model_name)
            pred = self._get_prediction(config, test_file, work_dir)
            predictions.append(pred)
            logger.info(f"Model {model_name} prediction completed")
        
        # Average predictions
        predictions = np.array(predictions)
        average_probs = np.mean(predictions, axis=0)
        flattened_probs = np.array(average_probs).flatten().tolist()
        prob_list = [f"{round(num, 3):.3f}" for num in flattened_probs]
        
        # Map SMILES to prediction results
        mols = getList(predictable_file)
        result = {mols[i]: prob_list[i] for i in range(len(prob_list))}
        
        logger.info(f"Ensemble voting completed for {len(result)} molecules")
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


# ==================== Toxicity Classifier ====================
class ToxicityClassifier:
    """Classify toxicity prediction based on probability thresholds"""
    
    # Classification thresholds (matching JavaScript logic)
    THRESHOLDS = {
        'highly_toxic': 0.5,           # prob >= 0.5
        'less_dili_concern': 0.38,     # prob > 0.38
        'safe': 0.12,                  # prob > 0.12
    }
    
    @staticmethod
    def classify(probability: float) -> str:
        """
        Classify DILI prediction based on probability value
        
        Args:
            probability: Probability value (float)
            
        Returns:
            Classification label (str)
        """
        if probability >= ToxicityClassifier.THRESHOLDS['highly_toxic']:
            return DILIPredictionClass.HIGHLY_TOXIC.value
        elif probability > ToxicityClassifier.THRESHOLDS['less_dili_concern']:
            return DILIPredictionClass.LESS_DILI_CONCERN.value
        elif probability > ToxicityClassifier.THRESHOLDS['safe']:
            return DILIPredictionClass.SAFE.value
        else:
            return DILIPredictionClass.HIGHLY_SAFE.value
    
    @staticmethod
    def classify_with_unpredictable(probability: Optional[float]) -> str:
        """
        Classify with unpredictable case handling
        
        Args:
            probability: Probability value (float) or None for unpredictable
            
        Returns:
            Classification label (str)
        """
        if probability is None:
            return DILIPredictionClass.UNPREDICTABLE.value
        return ToxicityClassifier.classify(probability)


# ==================== Result Writer ====================
class ResultWriter:

    @staticmethod
    def write_results(output_file: str, results: Dict[str, Tuple[str, str]]) -> None:
        """
        Write results to output file maintaining input order
        
        Args:
            output_file: Output file path
            results: Dictionary with SMILES as key and (probability, prediction) tuple as value
        """
        with open(output_file, 'w') as f:
            # Write header
            f.write("SMILES\tProbability\tPrediction\n")
            
            # Write results (preserving OrderedDict order)
            for smiles, (probability, prediction) in results.items():
                f.write(f"{smiles}\t{probability}\t{prediction}\n")
        
        logger.info(f"Results saved to {output_file}")


# ==================== DILI Predictor ====================
class DILIPredictor:
    """DILI prediction main class"""
    
    def __init__(self, work_dir: str = './', data_dir: str = './Dat'):
        self.work_dir = work_dir
        self.config = MolecularConfig()
        self.path_manager = PathManager(data_dir, work_dir)
        self.model_registry = ModelRegistry(data_dir)
        self.validator = SMILESValidator(self.config)
        self.selfies_converter = SELFIESConverter()
        self.embedding_generator = WordEmbeddingGenerator(self.path_manager)
        self.ensemble_voter = EnsembleVoter(self.model_registry, self.path_manager)
        self.toxicity_classifier = ToxicityClassifier()
        self.result_writer = ResultWriter()

    def _order_by_input_file(self, input_path: str,
                             result_dict: Dict[str, Tuple[str, str]]) -> OrderedDict:
        """Order results according to input file order"""
        ordered_result = OrderedDict()

        with open(input_path, 'r') as f:
            for line in f:
                smi = line.strip()
                if smi and smi in result_dict:
                    ordered_result[smi] = result_dict[smi]

        logger.info(f"Results ordered: {len(ordered_result)} molecules")
        return ordered_result

    def predict(self, input_filename: str = 'input.txt', 
                output_filename: str = 'output.txt',
                cleanup: bool = True) -> OrderedDict:

        input_path = self.work_dir + '/' + input_filename
        output_path = self.work_dir + '/' + output_filename
        
        try:
            # ===== Step 1: SMILES Validation =====
            logger.info("Step 1: Validating SMILES...")
            predictable_file = str(self.path_manager.get_predictable_file())
            unpredictable_file = str(self.path_manager.get_unpredictable_file())
            
            self.validator.validate_and_split(
                input_path, predictable_file, unpredictable_file
            )
            
            # Check if there are predictable molecules
            predictable_list = getList(predictable_file)
            unpredictable_list = getList(unpredictable_file)
            
            if len(predictable_list) == 0:
                logger.warning("No predictable molecules found")
                result = {}
                
                # Create results with unpredictable classification
                unpredictable_results = self._create_unpredictable_results(unpredictable_list)
                merge_result = MergeProcess(unpredictable_list, result)
                
                # Add classification for unpredictable molecules
                final_results = self._add_classification(merge_result)
                
                # Order results by input file
                ordered_result = self._order_by_input_file(input_path, final_results)
                
                # Save results
                self.result_writer.write_results(output_path, ordered_result)
                
                # Clean up temporary files
                if cleanup:
                    TemporaryFileManager.cleanup(self.work_dir)
                
                return ordered_result
            
            logger.info(f"Predictable: {len(predictable_list)}, "
                       f"Unpredictable: {len(unpredictable_list)}")
            
            # ===== Step 2: Calculate Mordred Descriptors =====
            logger.info("Step 2: Calculating Mordred descriptors...")
            calculateMord(predictable_file, self.work_dir)
            fset_mord = getFeatures(str(self.path_manager.get_feature_set_path('Mord')))
            saveSelectedWithToxicity(
                ['NAME'] + fset_mord,
                self.work_dir + '/Dsc_Mord.txt',
                self.work_dir + '/Dsc_Mord_fixed.txt'
            )
            
            # ===== Step 3: Convert SMILES to SELFIES =====
            logger.info("Step 3: Converting SMILES to SELFIES...")
            selfies_file = str(self.path_manager.get_selfies_file())
            self.selfies_converter.convert_file(predictable_file, selfies_file)
            
            # ===== Step 4: Calculate Dragon Descriptors =====
            logger.info("Step 4: Calculating Dragon descriptors...")
            configGenerator(self.work_dir, predictable_file)
            callDragon(self.work_dir)
            fset_dra = getFeatures(str(self.path_manager.get_feature_set_path('Dra')))
            saveSelectedWithToxicity(
                ['NAME'] + fset_dra,
                self.work_dir + '/Dsc_Dra.txt',
                self.work_dir + '/Dsc_Dra_fixed.txt'
            )
            
            # ===== Step 5: Merge Descriptors =====
            logger.info("Step 5: Merging features...")
            featureFusionner(self.work_dir)
            feature_set = getFeatures(str(self.path_manager.get_feature_set_path('X_DILiver')))
            saveSelectedWithToxicity(
                ['NAME'] + feature_set,
                self.work_dir + '/Fusion_Dra_Mord.txt',
                self.work_dir + '/DescFixedReduced.txt'
            )
            
            # ===== Step 6: Remove NA Rows =====
            logger.info("Step 6: Checking NA values...")
            NA_rowChecker(self.work_dir, predictable_file, unpredictable_file)
            
            # ===== Step 7: Generate TF-IDF Embeddings =====
            logger.info("Step 7: Generating TF-IDF embeddings...")
            self.embedding_generator.generate_embeddings(selfies_file, self.work_dir)
            
            # ===== Step 8: Ensemble Model Prediction =====
            logger.info("Step 8: Running ensemble prediction...")
            test_file = str(self.path_manager.get_descriptor_test_file())
            predictable_checked = str(self.path_manager.get_predictable_checked_file())
            
            result = self.ensemble_voter.vote_soft(
                test_file, predictable_checked, self.work_dir
            )

            # ===== Step 9: Merge Unpredictable Molecules =====
            logger.info("Step 9: Merging results...")
            unpredictable_list = getList(unpredictable_file)
            final_result = MergeProcess(unpredictable_list, result)

            # ===== Step 10: Add Classification =====
            logger.info("Step 10: Classifying toxicity predictions...")
            classified_results = self._add_classification(final_result)

            # ===== Step 11: Order Results by Input File =====
            logger.info("Step 11: Ordering results by input file...")
            ordered_result = self._order_by_input_file(input_path, classified_results)

            # ===== Step 12: Save Results =====
            logger.info("Step 12: Saving results...")
            self.result_writer.write_results(output_path, ordered_result)

            # ===== Step 13: Clean Up Temporary Files =====
            if cleanup:
                logger.info("Step 13: Cleaning up temporary files...")
                TemporaryFileManager.cleanup(self.work_dir)

            logger.info(f"Prediction completed for {len(ordered_result)} molecules")
            return ordered_result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise

    def _add_classification(self, results: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """
        Add toxicity classification to prediction results
        
        Args:
            results: Dictionary with SMILES as key and probability as value
            
        Returns:
            Dictionary with SMILES as key and (probability, classification) tuple as value
        """
        classified_results = {}
        
        for smiles, probability_str in results.items():
            if probability_str == "---":
                # Unpredictable molecule
                classification = DILIPredictionClass.UNPREDICTABLE.value
                classified_results[smiles] = (probability_str, classification)
            else:
                try:
                    probability = float(probability_str)
                    classification = self.toxicity_classifier.classify(probability)
                    classified_results[smiles] = (probability_str, classification)
                except ValueError:
                    logger.warning(f"Invalid probability value for {smiles}: {probability_str}")
                    classified_results[smiles] = (probability_str, DILIPredictionClass.UNPREDICTABLE.value)
        
        return classified_results

    def _create_unpredictable_results(self, unpredictable_list: List[str]) -> Dict[str, str]:
        """Create results for unpredictable molecules"""
        return {smiles: "---" for smiles in unpredictable_list}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='DILI Prediction Model - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input input.txt --output output.txt
  python main.py -i data/smiles.txt -o results/predictions.txt --data-dir ./Dat
        """
    )

    parser.add_argument(
        '-i', '--input',
        dest='input_file',
        required=True,
        help='Input file path (SMILES format)'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        required=True,
        help='Output file path for predictions'
    )

    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        default='./Dat',
        help='Data directory path (default: ./Dat)'
    )

    parser.add_argument(
        '--work-dir',
        dest='work_dir',
        default='./',
        help='Working directory path (default: ./)'
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

    logger.info(f"Starting DILI prediction...")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")

    # Verify input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize predictor
    predictor = DILIPredictor(work_dir=args.work_dir, data_dir=args.data_dir)

    # Set custom output file
    predictor.path_manager.set_output_file(args.output_file)

    # Run prediction
    try:
        result = predictor.predict(
            input_filename=input_path.name,
            output_filename=args.output_file,
            cleanup=True
        )
        logger.info(f"Prediction completed successfully!")
        logger.info(f"Results saved to: {args.output_file}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
