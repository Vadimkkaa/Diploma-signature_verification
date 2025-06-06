from core.signature_normalizer import SignatureNormalizer
from core.signature_features import SignatureFeaturesExtractor
from core.ocs_verifier import OCSVMVerifier
from utils.results_logger import ResultsLogger
from utils.user_registry import UserRegistry
import cv2
import os
import sqlite3


logger = ResultsLogger()
logger.view_results()