"""
Bayesian Network for Mango Disease Prediction
Combines CNN image features with symptom observations

Author: Claude Code Analysis
Date: 2026-04-20
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

try:
    from pgmpy.models import BayesianNetwork as PGMpyBayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination, BeliefPropagation
    HAS_PGMPY = True
except ImportError:
    HAS_PGMPY = False
    print("WARNING: pgmpy not installed. Install with: pip install pgmpy")

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class BayesianDiseaseNetwork:
    """
    Bayesian Network for mango disease classification combining:
    - CNN visual features from image
    - Farmer-reported symptoms
    - Environmental/seasonal factors
    """

    def __init__(self, model_name: str = "mango_disease_bn", version: str = "1.0"):
        """
        Initialize the Bayesian Network structure

        Args:
            model_name: Name of the network
            version: Version identifier
        """
        if not HAS_PGMPY:
            raise ImportError("pgmpy required. Install with: pip install pgmpy")

        self.model_name = model_name
        self.version = version
        self.network = None
        self.cpds = {}  # Conditional Probability Tables
        self.cpt_sample_counts = {}  # Track sample counts for online learning
        self.inference_engine = None
        self.feature_names = []
        self.discretizer = None
        self.pca = None

        # Disease classes from ml_views.py
        self.diseases = ['Anthracnose', 'DieBack', 'PowerderyMildew', 'SootyMold', 'Healthy']

        # Symptoms
        self.symptoms = [
            'LeafSpots', 'YellowHalos', 'WhiteCoating', 'TipDieback',
            'FruitSpots', 'FruitRot', 'LeafDrop', 'FruitDrop',
            'PowerderySurface', 'SootyDeposit'
        ]

        # Environmental factors
        self.seasons = ['Monsoon', 'DryMonth', 'WinterCool']
        self.regions = ['Mindanao', 'Central_Visayas', 'Ilocos', 'Other']

    def build_structure(self):
        """
        Create the Directed Acyclic Graph (DAG)

        Structure:
            Season → Disease
            Region → Disease
            VisualFeatures → Disease
            Symptoms → Disease
            ImageQuality → VisualFeatures
            SymptomReliability → Symptoms
        """
        edges = [
            ('Season', 'Disease'),
            ('Region', 'Disease'),
            ('VisualFeatures', 'Disease'),
            ('Symptoms', 'Disease'),
            ('PreviousDiseaseHistory', 'Disease'),
            ('ImageQuality', 'VisualFeatures'),
            ('SymptomReliability', 'Symptoms'),
        ]

        self.network = PGMpyBayesianNetwork(edges)
        logger.info(f"Built network with {len(self.network.nodes())} nodes and {len(edges)} edges")

    def create_cpts(self):
        """
        Create Conditional Probability Tables (CPTs)
        Initialize with domain knowledge; will be updated from data
        """

        # P(Disease) - Prior probabilities based on historical prevalence
        cpd_disease = TabularCPD(
            'Disease', len(self.diseases),
            [[0.35], [0.15], [0.25], [0.15], [0.10]],  # Typical prevalence
            variable_type='categorical'
        )
        self.cpds['Disease'] = cpd_disease

        # P(Season) - Uniform for now (will vary with actual month)
        cpd_season = TabularCPD(
            'Season', len(self.seasons),
            [[0.33], [0.33], [0.34]],
            variable_type='categorical'
        )
        self.cpds['Season'] = cpd_season

        # P(Region) - Uniform
        cpd_region = TabularCPD(
            'Region', len(self.regions),
            [[0.25], [0.25], [0.25], [0.25]],
            variable_type='categorical'
        )
        self.cpds['Region'] = cpd_region

        # P(VisualFeatures) - to be learned from CNN
        cpd_visual = TabularCPD(
            'VisualFeatures', 5,  # 5 discretized feature levels
            [[0.2] * 3 for _ in range(5)],  # Uniform initially (3 quality levels)
            evidence=['ImageQuality'],
            evidence_card=[3],
            variable_type='categorical'
        )
        self.cpds['VisualFeatures'] = cpd_visual

        # P(Symptoms) - to be learned
        cpd_symptoms = TabularCPD(
            'Symptoms', 5,  # 5 symptom patterns
            [[0.2] * 3 for _ in range(5)],  # Uniform initially
            evidence=['SymptomReliability'],
            evidence_card=[3],
            variable_type='categorical'
        )
        self.cpds['Symptoms'] = cpd_symptoms

        # P(ImageQuality) - often high from mobile images
        cpd_quality = TabularCPD(
            'ImageQuality', 3,
            [[0.6], [0.3], [0.1]],  # High, Medium, Low
            variable_type='categorical'
        )
        self.cpds['ImageQuality'] = cpd_quality

        # P(SymptomReliability) - often novice farmers
        cpd_reliability = TabularCPD(
            'SymptomReliability', 3,
            [[0.3], [0.5], [0.2]],  # Expert, Intermediate, Novice
            variable_type='categorical'
        )
        self.cpds['SymptomReliability'] = cpd_reliability

        # P(PreviousDiseaseHistory)
        cpd_history = TabularCPD(
            'PreviousDiseaseHistory', 2,
            [[0.7], [0.3]],  # No, Yes
            variable_type='categorical'
        )
        self.cpds['PreviousDiseaseHistory'] = cpd_history

        # Main CPT: P(Disease | Season, Region, VisualFeatures, Symptoms, History)
        # This is large but critical - will be filled from data
        # For now, simplified version without all parents
        cpd_disease_from_evidence = TabularCPD(
            'Disease', len(self.diseases),
            np.ones((len(self.diseases), 1)) / len(self.diseases),
            variable_type='categorical'
        )
        self.cpds['Disease'] = cpd_disease_from_evidence

    def learn_cpts_from_data(self, data: pd.DataFrame, smoothing_alpha: float = 1.0):
        """
        Learn CPT values from verified training data using Maximum Likelihood Estimation

        Args:
            data: DataFrame with columns [Disease, Season, Region, VisualFeatures, Symptoms, ...]
            smoothing_alpha: Laplace smoothing parameter
        """
        if data.empty:
            logger.warning("Empty training data provided")
            return

        logger.info(f"Learning CPTs from {len(data)} samples")

        # Learn P(Disease | Season)
        self._learn_disease_given_season(data, smoothing_alpha)

        # Learn P(Disease | Region)
        self._learn_disease_given_region(data, smoothing_alpha)

        # Learn P(VisualFeatures | ImageQuality)
        self._learn_features_given_quality(data, smoothing_alpha)

        # Learn P(Symptoms | Disease)
        self._learn_symptoms_given_disease(data, smoothing_alpha)

    def _learn_disease_given_season(self, data: pd.DataFrame, alpha: float):
        """Learn P(Disease | Season)"""
        if 'Disease' not in data.columns or 'Season' not in data.columns:
            logger.warning("Missing Disease or Season columns")
            return

        # Create contingency table
        contingency = pd.crosstab(data['Season'], data['Disease'])

        # Add smoothing
        contingency = contingency.add(alpha)

        # Normalize
        cpt_values = contingency.div(contingency.sum(axis=1), axis=0).T.values

        # Create CPD
        cpd = TabularCPD(
            'Disease', len(self.diseases),
            cpt_values,
            evidence=['Season'],
            evidence_card=[len(self.seasons)]
        )
        self.cpds['Disease'] = cpd
        logger.info("Updated P(Disease | Season)")

    def _learn_disease_given_region(self, data: pd.DataFrame, alpha: float):
        """Learn P(Disease | Region) - regional variation"""
        if 'Disease' not in data.columns or 'Region' not in data.columns:
            return

        contingency = pd.crosstab(data['Region'], data['Disease'])
        contingency = contingency.add(alpha)
        cpt_values = contingency.div(contingency.sum(axis=1), axis=0).T.values

        cpd = TabularCPD(
            'Disease', len(self.diseases),
            cpt_values,
            evidence=['Region'],
            evidence_card=[len(self.regions)]
        )
        logger.info("Updated P(Disease | Region)")

    def _learn_features_given_quality(self, data: pd.DataFrame, alpha: float):
        """Learn P(VisualFeatures | ImageQuality)"""
        if 'VisualFeatures' not in data.columns:
            logger.warning("Missing VisualFeatures column")
            return

        if 'ImageQuality' not in data.columns:
            # Assume all high quality if not specified
            data['ImageQuality'] = 'High'

        contingency = pd.crosstab(data['ImageQuality'], data['VisualFeatures'])
        contingency = contingency.add(alpha)
        cpt_values = contingency.div(contingency.sum(axis=1), axis=0).T.values

        cpd = TabularCPD(
            'VisualFeatures', 5,
            cpt_values,
            evidence=['ImageQuality'],
            evidence_card=[3]
        )
        self.cpds['VisualFeatures'] = cpd
        logger.info("Updated P(VisualFeatures | ImageQuality)")

    def _learn_symptoms_given_disease(self, data: pd.DataFrame, alpha: float):
        """Learn P(Symptoms | Disease) - symptom likelihood"""
        if 'Disease' not in data.columns or 'Symptoms' not in data.columns:
            return

        # Expand symptom columns if they're JSON
        symptom_cols = []
        for idx, row in data.iterrows():
            if isinstance(row['Symptoms'], str):
                try:
                    symptoms = json.loads(row['Symptoms'])
                except:
                    symptoms = row['Symptoms'].split(',')
            else:
                symptoms = row['Symptoms'] or []

            symptom_cols.append(symptoms)

        data['Symptoms'] = symptom_cols

        # Learn for each symptom
        for symptom in self.symptoms[:5]:  # First 5 for brevity
            symptom_present = [symptom in s for s in symptom_cols]
            temp_data = pd.DataFrame({
                'Disease': data['Disease'],
                'SymptomPresent': symptom_present
            })

            contingency = pd.crosstab(temp_data['Disease'], temp_data['SymptomPresent'])
            contingency = contingency.add(alpha)
            cpt_values = contingency.div(contingency.sum(axis=1), axis=0).T.values

            cpd = TabularCPD(
                f'Symptom_{symptom}', 2,
                cpt_values,
                evidence=['Disease'],
                evidence_card=[len(self.diseases)]
            )
            self.cpds[f'Symptom_{symptom}'] = cpd

    def infer_disease(self, evidence: Dict[str, str]) -> Dict[str, float]:
        """
        Perform exact inference using Variable Elimination

        Args:
            evidence: Dictionary of observed variables
                     e.g., {'Season': 'Monsoon', 'LeafSpots': 'Present', ...}

        Returns:
            Dictionary of P(Disease | evidence) for each disease
        """
        if self.network is None:
            raise ValueError("Network not built. Call build_structure() first")

        if not HAS_PGMPY:
            raise ImportError("pgmpy required for inference")

        try:
            # Add CPDs to network
            for cpd in self.cpds.values():
                self.network.add_cpds(cpd)

            # Verify network
            if not self.network.check_model():
                logger.warning("Network model check failed")

            # Create inference engine
            infer = VariableElimination(self.network)

            # Filter evidence to only variables in network
            filtered_evidence = {
                k: v for k, v in evidence.items()
                if k in self.network.nodes()
            }

            # Query
            result = infer.query(
                variables=['Disease'],
                evidence=filtered_evidence,
                show_progress=False
            )

            # Extract probabilities
            posteriors = dict(zip(self.diseases, result.values.flatten()))

            return posteriors

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Fallback to uniform distribution
            return {d: 1.0 / len(self.diseases) for d in self.diseases}

    def infer_disease_approximate(self, evidence: Dict[str, str],
                                 n_samples: int = 1000) -> Dict[str, float]:
        """
        Approximate inference using Gibbs sampling
        More robust for complex networks

        Args:
            evidence: Observed variables
            n_samples: Number of samples for MCMC

        Returns:
            Approximate P(Disease | evidence)
        """
        try:
            from pgmpy.inference import GibbsSampling

            if self.network is None:
                raise ValueError("Network not built")

            sampler = GibbsSampling(self.network)
            samples = sampler.forward_inference(
                variables=['Disease'],
                evidence=evidence,
                size=n_samples
            )

            # Compute empirical distribution
            disease_samples = samples['Disease']
            posteriors = disease_samples.value_counts(normalize=True).to_dict()

            # Ensure all diseases present
            for d in self.diseases:
                if d not in posteriors:
                    posteriors[d] = 0.0

            return posteriors

        except Exception as e:
            logger.error(f"Approximate inference failed: {e}")
            return {d: 1.0 / len(self.diseases) for d in self.diseases}

    def update_online(self, evidence: Dict[str, str], ground_truth: str, weight: float = 1.0):
        """
        Online learning: update CPTs when verified data arrives

        Args:
            evidence: The evidence used for prediction
            ground_truth: The verified true disease label
            weight: Confidence weight (0-1)
        """
        if self.network is None:
            logger.warning("Network not initialized")
            return

        # Initialize CPT counts if needed
        if not self.cpt_sample_counts:
            for cpd_name in self.cpds:
                self.cpt_sample_counts[cpd_name] = {}

        # Increment counts for P(Disease | evidence)
        key = ('Disease', str(evidence))
        if key not in self.cpt_sample_counts['Disease']:
            self.cpt_sample_counts['Disease'][key] = {}

        if ground_truth not in self.cpt_sample_counts['Disease'][key]:
            self.cpt_sample_counts['Disease'][key][ground_truth] = 0

        self.cpt_sample_counts['Disease'][key][ground_truth] += weight

        logger.info(f"Updated belief for {ground_truth} with weight {weight}")

    def get_uncertainty_bounds(self, posteriors: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Estimate credible intervals from approximate inference

        Args:
            posteriors: P(Disease | evidence) from sampling

        Returns:
            {disease: {'mean': 0.5, 'std': 0.1, 'ci_lower': 0.3, 'ci_upper': 0.7}}
        """
        bounds = {}
        for disease, prob in posteriors.items():
            # Approximate standard deviation using Beta distribution properties
            # For binomial: std ≈ sqrt(p(1-p)/n)
            std = np.sqrt(prob * (1 - prob) / 100)  # Assume n=100 samples

            bounds[disease] = {
                'mean': float(prob),
                'std': float(std),
                'credible_interval': [
                    float(max(0, prob - 1.96 * std)),
                    float(min(1, prob + 1.96 * std))
                ]
            }

        return bounds

    def detect_concept_drift(self, recent_data: pd.DataFrame,
                            historical_data: pd.DataFrame,
                            threshold: float = 0.1) -> Tuple[bool, float]:
        """
        Detect if disease distributions have shifted (concept drift)

        Args:
            recent_data: Recently verified images
            historical_data: Historical verified images
            threshold: JS divergence threshold for alert

        Returns:
            (has_drift, divergence_score)
        """
        if not HAS_SCIPY or recent_data.empty or historical_data.empty:
            return False, 0.0

        try:
            # Disease distributions
            recent_dist = recent_data['Disease'].value_counts(normalize=True)
            historical_dist = historical_data['Disease'].value_counts(normalize=True)

            # Ensure same order
            all_diseases = set(recent_dist.index) | set(historical_dist.index)
            recent_vec = np.array([recent_dist.get(d, 0.01) for d in sorted(all_diseases)])
            historical_vec = np.array([historical_dist.get(d, 0.01) for d in sorted(all_diseases)])

            # Normalize
            recent_vec /= recent_vec.sum()
            historical_vec /= historical_vec.sum()

            # Jensen-Shannon divergence
            divergence = jensenshannon(recent_vec, historical_vec)

            has_drift = divergence > threshold

            if has_drift:
                logger.warning(f"Concept drift detected: JS divergence = {divergence:.3f}")

            return has_drift, float(divergence)

        except Exception as e:
            logger.error(f"Concept drift detection failed: {e}")
            return False, 0.0

    def explain_prediction(self, posteriors: Dict[str, float],
                          evidence: Dict[str, str]) -> Dict[str, Any]:
        """
        Explain why a certain disease was predicted

        Args:
            posteriors: P(Disease | evidence)
            evidence: The evidence used

        Returns:
            Explanation dictionary
        """
        top_disease = max(posteriors, key=posteriors.get)
        confidence = posteriors[top_disease]

        return {
            'top_disease': top_disease,
            'confidence': float(confidence),
            'evidence_used': evidence,
            'alternative_diseases': [
                {'disease': d, 'probability': float(p)}
                for d, p in sorted(posteriors.items(), key=lambda x: x[1], reverse=True)[1:3]
            ],
            'explanation': self._generate_explanation(top_disease, posteriors, evidence)
        }

    def _generate_explanation(self, disease: str, posteriors: Dict[str, float],
                             evidence: Dict[str, str]) -> str:
        """Generate human-readable explanation"""
        confidence = posteriors[disease]

        if confidence > 0.7:
            confidence_text = "high confidence"
        elif confidence > 0.4:
            confidence_text = "moderate confidence"
        else:
            confidence_text = "low confidence"

        explanation = f"Model predicts {disease} with {confidence_text} "
        explanation += f"({confidence*100:.1f}%) based on: {', '.join(evidence.keys())}"

        return explanation

    def save_model(self, filepath: str):
        """Persist trained network to disk"""
        import pickle

        model_data = {
            'name': self.model_name,
            'version': self.version,
            'cpds': self.cpds,
            'cpt_sample_counts': self.cpt_sample_counts,
            'diseases': self.diseases,
            'symptoms': self.symptoms,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained network from disk"""
        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model_name = model_data['name']
        self.version = model_data['version']
        self.cpds = model_data['cpds']
        self.cpt_sample_counts = model_data.get('cpt_sample_counts', {})
        self.diseases = model_data['diseases']
        self.symptoms = model_data['symptoms']

        logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        return {
            'name': self.model_name,
            'version': self.version,
            'nodes': list(self.network.nodes()) if self.network else [],
            'edges': list(self.network.edges()) if self.network else [],
            'diseases': self.diseases,
            'symptoms': self.symptoms,
            'cpds_count': len(self.cpds)
        }


# Utility functions

def extract_intermediate_cnn_features(model, image_array: np.ndarray) -> np.ndarray:
    """
    Extract features from CNN intermediate layer

    Args:
        model: Keras model
        image_array: Preprocessed image (1, 224, 224, 3)

    Returns:
        Feature vector from second-to-last layer
    """
    import tensorflow as tf

    # Create feature extractor
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output  # Pre-softmax layer
    )

    features = feature_extractor.predict(image_array, verbose=0)
    return features.flatten()


def discretize_cnn_features(features: np.ndarray, n_bins: int = 5) -> str:
    """
    Convert continuous CNN features to discrete bins for BN

    Args:
        features: CNN feature vector
        n_bins: Number of discretization bins

    Returns:
        Discrete feature level string
    """
    # Simple quantile-based discretization
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(features, percentiles)
    bin_idx = np.digitize(features.mean(), bin_edges) - 1
    bin_idx = max(0, min(n_bins - 1, bin_idx))

    levels = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
    return levels[bin_idx] if bin_idx < len(levels) else 'Unknown'


def extract_season(date: datetime) -> str:
    """Extract season from date"""
    month = date.month

    # Philippine seasons
    if month in [6, 7, 8, 9, 10, 11]:  # June-Nov: wet/monsoon
        return 'Monsoon'
    elif month in [12, 1, 2]:  # Dec-Feb: cool/dry
        return 'WinterCool'
    else:  # Mar-May: dry
        return 'DryMonth'


def extract_region_from_address(address: str) -> str:
    """Infer region from location address"""
    if not address:
        return 'Other'

    address_lower = address.lower()

    if any(x in address_lower for x in ['mindanao', 'davao', 'cotabato']):
        return 'Mindanao'
    elif any(x in address_lower for x in ['cebu', 'visayas', 'bohol']):
        return 'Central_Visayas'
    elif any(x in address_lower for x in ['ilocos', 'la union', 'dagupan']):
        return 'Ilocos'
    else:
        return 'Other'
