# Configuration constants for loqculate.
# Centralizes every configurable value in one place.

DEFAULT_STD_MULT = 2  # multiplier of noise std for LOD
DEFAULT_CV_THRESH = 0.2  # 20 % CV threshold for LOQ
DEFAULT_BOOT_REPS = 100  # number of bootstrap replicates
DEFAULT_LOQ_GRID_POINTS = 100  # grid bins between LOD and max(x)
DEFAULT_MIN_NOISE_POINTS = 2  # minimum calibration points below LOD
DEFAULT_MIN_LINEAR_POINTS = 1  # minimum calibration points above LOD
DEFAULT_WEIGHT_CAP = 1000  # max value for WLS weight 1/sqrt(x)
DEFAULT_SLIDING_WINDOW = 3  # consecutive points that must stay below CV threshold
DEFAULT_CHUNK_SIZE = 100  # peptides per multiprocessing batch

KNOT_SEARCH_SINGULAR_THRESHOLD = 1e-30  # min |det(G)| before falling back to weighted mean
