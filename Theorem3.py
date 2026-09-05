"""
Theorem 3 Validation: Model-Independent Failure Criterion

This script validates Theorem 3:
When the geometric condition is violated (G <= C_2), 
ALL algorithms asymptotically collapse to random guessing:
AUC(A) -> 1/2, for all A.

Validation strategy (Paper Section 3):
1. Reverse validation: Use T1a low-GDI extreme configurations
2. For each dataset, identify the lowest-GDI configuration (largest n)
3. Run W1-Detector at this configuration
4. Perform one-sample t-test comparing AUC to 0.5
5. Report AUC, 95% CI, and p-value

All results are saved to ./results4/
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from scipy.integrate import simpson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Global Configuration
# ============================================================================

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

ALPHA = 0.1
M = 2
N_REPEATS = 30
SIGNIFICANCE_LEVEL = 0.05

P_FIXED = {'AD': 80, 'IIoT': 10, 'Finance': 50}

N_RANGE = {
    'AD': [200, 400, 800, 1200, 2000],
    'IIoT': [500, 1000, 2000, 5000, 10000],
    'Finance': [1000, 3000, 5000, 10000, 30000]
}

COLORS = {'AD': '#E41A1C', 'IIoT': '#377EB8', 'Finance': '#FF7F00'}

RESULTS_DIR = './results4'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# Data Loaders (same as previous)
# ============================================================================

class ADImageLoader:
    def __init__(self, data_root, normal_root='NonDemented', anomaly_root='ModerateDemented',
                 target_size=(32, 32), sample_limit=None):
        from PIL import Image
        self.data_root = data_root
        self.normal_root = normal_root
        self.anomaly_root = anomaly_root
        self.target_size = target_size
        self.sample_limit = sample_limit
        self.Image = Image
        
    def _load_from_folder(self, folder_path, label, prefix_filter=None):
        X, y = [], []
        if not os.path.isdir(folder_path):
            print(f"Warning: folder {folder_path} not found, skipped.")
            return X, y
        print(f"Loading images from: {folder_path}")
        for root, _, files in os.walk(folder_path):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    continue
                if prefix_filter and not file.startswith(prefix_filter):
                    continue
                img_path = os.path.join(root, file)
                try:
                    img = self.Image.open(img_path).convert('L')
                    img = img.resize(self.target_size)
                    arr = np.array(img).flatten() / 255.0
                    X.append(arr)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        return X, y
    
    def load_data(self):
        X_all, y_all = [], []
        normal_path = os.path.join(self.data_root, self.normal_root)
        X_normal, y_normal = self._load_from_folder(normal_path, 0, prefix_filter='nonDem')
        X_all.extend(X_normal)
        y_all.extend(y_normal)
        print(f"Loaded {len(X_normal)} normal images")
        anomaly_path = os.path.join(self.data_root, self.anomaly_root)
        X_anomaly, y_anomaly = self._load_from_folder(anomaly_path, 1, prefix_filter='moderateDem')
        X_all.extend(X_anomaly)
        y_all.extend(y_anomaly)
        print(f"Loaded {len(X_anomaly)} anomaly images")
        if len(X_all) == 0:
            raise ValueError(f"No images loaded! Check paths: {self.data_root}")
        X = np.array(X_all)
        y = np.array(y_all)
        print(f"\n=== Dataset Statistics ===")
        print(f"Total images: {len(X)}")
        print(f"Normal images: {np.sum(y == 0)}")
        print(f"Anomaly images: {np.sum(y == 1)}")
        print(f"Anomaly rate: {np.mean(y):.4%}")
        return X, y


class IIoTDataLoader:
    def __init__(self, random_state=42):
        self.scaler = StandardScaler()
        self.random_state = random_state
        
    def load_and_preprocess_data(self):
        try:
            print("Loading IIoT data...")
            data = pd.read_csv('./Datasets/IOT/UKMNCT_IIoT_FDIA.csv', header=0)
            X = data.iloc[:, :-1]
            y_raw = data.iloc[:, -1]
            print(f"Unique label values: {y_raw.unique()}")
            y = y_raw.map({'Attack': 1, 'Natural': 0})
            if y.isna().any():
                if y_raw.dtype in ['int64', 'float64']:
                    y = y_raw.copy()
                else:
                    y = y_raw.apply(lambda x: 1 if str(x).lower() in ['attack', '1', 'anomaly'] else 0)
            X_ = X.values.astype(float)
            y_ = y.values.flatten().astype(int)
            print(f"Data shape: {X_.shape}, Label shape: {y_.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        print("Performing data standardization...")
        X_scaled = self.scaler.fit_transform(X_)
        print(f"Anomaly rate: {np.mean(y_):.4%}")
        return X_scaled, y_


class FinanceDataLoader:
    def __init__(self, random_state=42):
        self.scaler = StandardScaler()
        self.random_state = random_state
        
    def load_and_preprocess_data(self):
        try:
            print("Loading Finance data...")
            _data = pd.read_csv('./Datasets/Finance/FinanceData.csv', skiprows=1, header=None)
            data = _data.iloc[:, :-2]
            labels = pd.read_csv('./Datasets/Finance/FinanceLabel.csv', skiprows=1, header=None)
            X_ = data.values.astype(float)
            y_ = labels.values.flatten().astype(int)
            print(f"Data shape: {X_.shape}, Label shape: {y_.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        print("Performing data standardization...")
        X_scaled = self.scaler.fit_transform(X_)
        print(f"Anomaly rate: {np.mean(y_):.4%}")
        return X_scaled, y_


# ============================================================================
# Spectral Utility Functions
# ============================================================================

def compute_eigenvalue_spacings(X, alpha=ALPHA, m=M):
    n, p = X.shape
    if n < 2 or p < 2:
        return np.array([])
    
    X_centered = X - np.mean(X, axis=0)
    Sigma_hat = (X_centered.T @ X_centered) / n
    
    eigvals = np.linalg.eigvalsh(Sigma_hat)
    eigvals = np.sort(eigvals)[::-1]
    
    start_idx = int(np.floor(alpha * p))
    end_idx = int(np.floor((1 - alpha) * p))
    
    if end_idx - start_idx < 2:
        return np.array([])
    
    bulk_eigvals = eigvals[start_idx:end_idx]
    raw_spacings = np.diff(bulk_eigvals)
    
    if len(raw_spacings) < 2:
        return np.array([])
    
    smoothed_spacings = []
    for j in range(len(raw_spacings)):
        j_start = max(0, j - m)
        j_end = min(len(raw_spacings), j + m + 1)
        delta_j = np.mean(raw_spacings[j_start:j_end])
        if delta_j > 1e-10:
            smoothed_spacings.append(raw_spacings[j] / delta_j)
    
    return np.array(smoothed_spacings)


def compute_w1_distance(spacings, target='wd'):
    if len(spacings) < 3:
        return 0.0
    
    s_sorted = np.sort(spacings)
    f_emp = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
    
    s_max = max(10.0, s_sorted.max() * 1.2)
    s_grid = np.linspace(0, s_max, 5000)
    f_emp_interp = np.interp(s_grid, s_sorted, f_emp, left=0, right=1)
    
    if target == 'wd':
        f_theory = 1 - np.exp(-np.pi * s_grid**2 / 4)
    else:
        f_theory = 1 - np.exp(-s_grid)
    
    diff = np.abs(f_emp_interp - f_theory)
    return simpson(diff, s_grid)


def compute_delta(spacings):
    if len(spacings) < 3:
        return 0.0
    d_wd = compute_w1_distance(spacings, 'wd')
    d_p = compute_w1_distance(spacings, 'p')
    return d_wd - d_p


def compute_gdi(tau, kappa, sigma2, p, n):
    if kappa <= 0 or sigma2 <= 0 or n <= 0:
        return 0.0
    return (tau**2 * p) / (kappa**2 * sigma2 * n)


def w1_detector_score(X_normal, X_test, alpha=ALPHA, m=M):
    """
    W1-Detector: T(x) = d_P(x) - d_WD(x)
    Positive score: closer to WD (normal-like)
    Negative score: closer to Poisson (anomaly-like)
    """
    n_normal = X_normal.shape[0]
    n_test = X_test.shape[0]
    scores = np.zeros(n_test)
    
    if n_normal < 3 or n_test < 1:
        return scores
    
    for i in range(n_test):
        X_combined = np.vstack([X_normal, X_test[i:i+1]])
        spacings = compute_eigenvalue_spacings(X_combined, alpha, m)
        if len(spacings) < 3:
            scores[i] = 0.0
            continue
        d_wd = compute_w1_distance(spacings, 'wd')
        d_p = compute_w1_distance(spacings, 'p')
        scores[i] = d_p - d_wd
    
    return scores


# ============================================================================
# Geometry Estimation (Section 3.4, Niyogi et al. [26])
# ============================================================================

class GeometryEstimator:
    def __init__(self, k_neighbors=30):
        self.k_neighbors = k_neighbors
        self.tau = None
        self.kappa = None
        self.sigma2 = None
        self.d = None
    
    def estimate(self, X):
        n, D = X.shape
        if n < 10 or D < 2:
            self.tau, self.kappa, self.sigma2, self.d = 1.0, 1.0, 1e-6, 2
            return self.tau, self.kappa, self.sigma2, self.d
        
        self._estimate_intrinsic_dimension(X)
        self.tau = self._estimate_reach(X)
        self.kappa = self._estimate_curvature(X)
        self.sigma2 = self._estimate_noise_variance(X)
        return self.tau, self.kappa, self.sigma2, self.d
    
    def _estimate_intrinsic_dimension(self, X):
        n, D = X.shape
        if D <= 1:
            self.d = 1
            return
        pca = PCA()
        pca.fit(X)
        eigvals = pca.explained_variance_
        cumsum = np.cumsum(eigvals) / np.sum(eigvals)
        d = np.argmax(cumsum >= 0.90) + 1
        self.d = max(1, min(d, D))
    
    def _estimate_reach(self, X):
        n, D = X.shape
        k = min(self.k_neighbors, n)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        reach_estimates = []
        sample_size = min(n, 200)
        sample_indices = np.random.choice(n, sample_size, replace=False)
        
        for i in sample_indices:
            neighborhood = X[indices[i]]
            k_local = len(neighborhood)
            if k_local < 5:
                continue
            
            mu = np.mean(neighborhood, axis=0)
            centered = neighborhood - mu
            cov_local = centered.T @ centered / k_local
            eigvals = np.linalg.eigvalsh(cov_local)
            eigvals = np.sort(eigvals)[::-1]
            
            d = min(self.d, len(eigvals) - 1)
            if d >= 1 and d < len(eigvals):
                tangent_var = np.sum(eigvals[:d])
                normal_var = np.sum(eigvals[d:])
                if normal_var > 1e-10:
                    ratio = tangent_var / normal_var
                    local_scale = np.median(distances[i, 1:min(5, k_local)])
                    reach_est = local_scale * np.sqrt(ratio)
                    reach_estimates.append(reach_est)
        
        if reach_estimates:
            return np.median(reach_estimates)
        return 1.0
    
    def _estimate_curvature(self, X):
        n, D = X.shape
        k = min(self.k_neighbors, n)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        curvature_estimates = []
        sample_size = min(n, 200)
        sample_indices = np.random.choice(n, sample_size, replace=False)
        
        for i in sample_indices:
            neighborhood = X[indices[i]]
            k_local = len(neighborhood)
            if k_local < 5:
                continue
            
            mu = np.mean(neighborhood, axis=0)
            centered = neighborhood - mu
            cov_local = centered.T @ centered / k_local
            eigvals = np.linalg.eigvalsh(cov_local)
            eigvals = np.sort(eigvals)[::-1]
            
            if len(eigvals) >= 3 and eigvals[0] > 1e-10:
                r = np.median(distances[i, 1:min(3, k_local)])
                if r > 1e-10:
                    ratio = np.sqrt(eigvals[1] + eigvals[2]) / np.sqrt(np.sum(eigvals[:3]))
                    kappa_est = ratio / r
                    kappa_est = max(0.001, min(100.0, kappa_est))
                    curvature_estimates.append(kappa_est)
        
        if curvature_estimates:
            return np.max(curvature_estimates)
        return 1.0
    
    def _estimate_noise_variance(self, X):
        n, D = X.shape
        d = min(self.d, D)
        if d < D and d > 0:
            pca = PCA(n_components=d)
            X_proj = pca.fit_transform(X)
            X_recon = pca.inverse_transform(X_proj)
            residual = X - X_recon
            sigma2 = np.var(residual.flatten())
        else:
            sigma2 = np.var(X.flatten())
        return max(sigma2, 1e-6)


# ============================================================================
# Find Lowest-GDI Configuration (from T1a)
# ============================================================================

def find_lowest_gdi_config(X_normal, dataset_name, p_fixed, n_values):
    """
    Find the lowest-GDI configuration from T1a subsampling.
    """
    print(f"\n  Finding lowest-GDI configuration for {dataset_name}...")
    
    n_normal_full = len(X_normal)
    
    geo_estimator = GeometryEstimator()
    tau, kappa, sigma2, d = geo_estimator.estimate(X_normal)
    print(f"    Geometry: tau={tau:.4f}, kappa={kappa:.4f}, sigma2={sigma2:.6f}, d={d}")
    
    valid_ns = [n for n in n_values if n <= n_normal_full]
    if not valid_ns:
        print(f"    No valid n values for {dataset_name}")
        return None
    
    n_lowest_gdi = max(valid_ns)
    gdi = compute_gdi(tau, kappa, sigma2, p_fixed, n_lowest_gdi)
    
    print(f"    Lowest-GDI configuration: n={n_lowest_gdi}, GDI={gdi:.6f}")
    
    return {
        'dataset': dataset_name,
        'n': n_lowest_gdi,
        'p': p_fixed,
        'tau': tau,
        'kappa': kappa,
        'sigma2': sigma2,
        'gdi': gdi,
        'd': d
    }


# ============================================================================
# Run W1-Detector on Low-GDI Configuration (Theorem 4)
# ============================================================================

def run_theorem4(X_normal, X_anomaly, dataset_name, config, n_repeats=30):
    """
    Run W1-Detector on the low-GDI configuration.
    
    Paper Section 3.1.5:
    - Compute AUC of W1-detector at lowest-GDI configuration
    - Perform one-sample t-test comparing AUC to 0.5
    - Report AUC, 95% CI, and p-value
    """
    print(f"\n  Running W1-Detector on low-GDI configuration for {dataset_name}...")
    
    n = config['n']
    p = config['p']
    n_normal_full = len(X_normal)
    n_anomaly_full = len(X_anomaly)
    
    aucs = []
    
    for rep in range(n_repeats):
        # Draw n normal samples for training (low-GDI = large n)
        idx_train = np.random.choice(n_normal_full, n, replace=False)
        X_train = X_normal[idx_train]
        
        # Test: remaining normal samples + ALL anomalies (no overlap)
        idx_test = np.array([i for i in range(n_normal_full) if i not in idx_train])
        X_test_normal = X_normal[idx_test]
        X_test_anomaly = X_anomaly
        
        # Project to fixed p (fit on training only)
        p_eff = min(p, X_train.shape[1])
        if p_eff <= 0:
            continue
        
        pca = PCA(n_components=p_eff)
        X_train_proj = pca.fit_transform(X_train)
        X_test_normal_proj = pca.transform(X_test_normal)
        X_test_anomaly_proj = pca.transform(X_anomaly)
        
        # Pad if needed
        if p_eff < p:
            X_train_proj = np.hstack([X_train_proj, np.zeros((n, p - p_eff))])
            X_test_normal_proj = np.hstack([X_test_normal_proj, np.zeros((len(X_test_normal), p - p_eff))])
            X_test_anomaly_proj = np.hstack([X_test_anomaly_proj, np.zeros((n_anomaly_full, p - p_eff))])
        
        # Combine test data
        X_test_all = np.vstack([X_test_normal_proj, X_test_anomaly_proj])
        y_test_all = np.concatenate([np.zeros(len(X_test_normal)), np.ones(n_anomaly_full)])
        
        # W1-Detector
        scores = w1_detector_score(X_train_proj, X_test_all)
        
        try:
            auc = roc_auc_score(y_test_all, scores)
        except:
            auc = 0.5
        
        # For AUC < 0.5, use 1 - AUC (equivalent to inverted classifier)
        # Both indicate random guessing
        auc = max(auc, 1 - auc)
        aucs.append(auc)
    
    # Statistics
    if aucs:
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        ci_lower = np.percentile(aucs, 2.5)
        ci_upper = np.percentile(aucs, 97.5)
        
        # Handle the case where all AUCs are exactly 0.5 (zero variance)
        # t-test is undefined when variance is zero
        if auc_std == 0 and auc_mean == 0.5:
            t_stat = 0.0
            p_value = 1.0
            is_random = True
        else:
            t_stat, p_value = ttest_1samp(aucs, 0.5)
            is_random = (p_value >= SIGNIFICANCE_LEVEL)
        
        result = {
            'dataset': dataset_name,
            'detector': 'W1-Detector',
            'n': n,
            'p': p,
            'gdi': config['gdi'],
            'tau': config['tau'],
            'kappa': config['kappa'],
            'sigma2': config['sigma2'],
            'auc_mean': auc_mean,
            'auc_std': auc_std,
            'auc_ci_lower': ci_lower,
            'auc_ci_upper': ci_upper,
            't_stat': t_stat,
            'p_value': p_value,
            'is_random_guessing': is_random,
            'n_replicates': len(aucs)
        }
        
        status = "RANDOM (Theorem 4 supported)" if is_random else "DETECTABLE (Theorem 4 NOT supported)"
        print(f"    AUC={auc_mean:.4f}±{auc_std:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}], "
              f"p={p_value:.4e}, {status}")
        
        return result
    else:
        print(f"    No valid results for {dataset_name}")
        return None


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_figure_4(df_results):
    """
    Generate Figure 4: Bar plot of AUC for W1-Detector at low-GDI configurations.
    
    One subplot: each dataset as a bar with error bar.
    Horizontal line at 0.5 (random guessing).
    """
    print("\nGenerating Figure 4...")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    datasets = df_results['dataset'].values
    auc_vals = df_results['auc_mean'].values
    auc_stds = df_results['auc_std'].values
    p_values = df_results['p_value'].values
    ci_lower = df_results['auc_ci_lower'].values
    ci_upper = df_results['auc_ci_upper'].values
    gdi_vals = df_results['gdi'].values
    
    x = np.arange(len(datasets))
    colors = [COLORS[d] for d in datasets]
    
    # Bar plot
    bars = ax.bar(x, auc_vals, yerr=auc_stds, color=colors,
                  edgecolor='black', linewidth=1.5, capsize=5, alpha=0.8)
    
    # Add labels above bars
    for i, (bar, p_val, ci_l, ci_u, gdi) in enumerate(zip(bars, p_values, ci_lower, ci_upper, gdi_vals)):
        # p-value text
        p_text = f'p={p_val:.3f}' if p_val < 0.001 else f'p={p_val:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               p_text, ha='center', va='bottom', fontsize=10)
        # CI text
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
               f'95% CI [{ci_l:.3f}, {ci_u:.3f}]', ha='center', va='top', fontsize=8)
        # GDI text below x-axis
        ax.text(bar.get_x() + bar.get_width()/2, -0.06,
               f'GDI={gdi:.4f}', ha='center', va='top', fontsize=8, rotation=0)
    
    # Random guessing baseline
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Random guessing (0.5)')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('AUC (W1-Detector)', fontsize=12)
    ax.set_title('Theorem 4: Model-Independent Failure Criterion\nLowest-GDI Configuration',
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.3, 0.7)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, 'Figure4_Theorem4.tiff')
    plt.savefig(output_path, format='tiff', dpi=150, bbox_inches='tight')
    print(f"Figure 4 saved to: {output_path}")
    
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 70)
    print("THEOREM 4 VALIDATION: MODEL-INDEPENDENT FAILURE CRITERION")
    print("=" * 70)
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Significance level: {SIGNIFICANCE_LEVEL}")
    print("Using only W1-Detector (per paper Section 3.1.5)")
    
    dataset_configs = {
        'AD': {
            'loader': ADImageLoader,
            'loader_kwargs': {'data_root': './Datasets/AD', 'normal_root': 'NonDemented',
                            'anomaly_root': 'ModerateDemented', 'target_size': (32, 32)},
            'p_fixed': P_FIXED['AD'],
            'n_range': N_RANGE['AD'],
        },
        'IIoT': {
            'loader': IIoTDataLoader,
            'loader_kwargs': {},
            'p_fixed': P_FIXED['IIoT'],
            'n_range': N_RANGE['IIoT'],
        },
        'Finance': {
            'loader': FinanceDataLoader,
            'loader_kwargs': {},
            'p_fixed': P_FIXED['Finance'],
            'n_range': N_RANGE['Finance'],
        }
    }
    
    all_results = []
    
    for dataset_name, config in dataset_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        loader = config['loader'](**config['loader_kwargs'])
        
        if dataset_name == 'AD':
            X_full, y_full = loader.load_data()
        else:
            data = loader.load_and_preprocess_data()
            if data is None:
                print(f"Failed to load {dataset_name}, skipping...")
                continue
            X_full, y_full = data
        
        X_normal = X_full[y_full == 0]
        X_anomaly = X_full[y_full == 1]
        
        n_normal_full = len(X_normal)
        n_anomaly_full = len(X_anomaly)
        n_features = X_full.shape[1]
        anomaly_rate = n_anomaly_full / (n_normal_full + n_anomaly_full) if (n_normal_full + n_anomaly_full) > 0 else 0
        
        print(f"\n{dataset_name} Summary:")
        print(f"  Total samples: {n_normal_full + n_anomaly_full}, Features: {n_features}")
        print(f"  Normal: {n_normal_full}, Anomaly: {n_anomaly_full} ({anomaly_rate:.4%})")
        
        # Find lowest-GDI configuration
        low_gdi_config = find_lowest_gdi_config(
            X_normal, dataset_name,
            config['p_fixed'], config['n_range']
        )
        
        if low_gdi_config is None:
            print(f"  No valid low-GDI configuration for {dataset_name}, skipping...")
            continue
        
        # Run W1-Detector on low-GDI configuration
        result = run_theorem4(
            X_normal, X_anomaly, dataset_name,
            low_gdi_config, n_repeats=N_REPEATS
        )
        
        if result is not None:
            all_results.append(result)
    
    # Combine results
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(os.path.join(RESULTS_DIR, 'summary4.csv'), index=False)
        
        # Generate Figure 4
        plot_figure_4(df_all)
        
        print("\n" + "=" * 70)
        print("THEOREM 4 VALIDATION COMPLETE!")
        print("=" * 70)
        print(f"Results saved in: {RESULTS_DIR}")
        print("  - summary4.csv: W1-Detector results at lowest-GDI configurations")
        print("  - Figure4_Theorem4.tiff: Bar plot with p-values and 95% CIs")
        
        # Verification Summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY (Theorem 4)")
        print("=" * 70)
        
        all_random = True
        for _, row in df_all.iterrows():
            dataset = row['dataset']
            gdi = row['gdi']
            auc = row['auc_mean']
            p_val = row['p_value']
            is_random = row['is_random_guessing']
            
            status = "✓ RANDOM (Theorem 4 supported)" if is_random else "✗ DETECTABLE (Theorem 4 NOT supported)"
            print(f"  {dataset}: GDI={gdi:.6f}, AUC={auc:.4f}, p={p_val:.4e} -> {status}")
            
            if not is_random:
                all_random = False
        
        if all_random:
            print("\n  ✓ ALL datasets support Theorem 4: AUC is statistically indistinguishable from 0.5")
        else:
            print("\n  ✗ Some datasets do NOT support Theorem 4")
        
        print("\n" + "=" * 70)
    else:
        print("No results generated!")


if __name__ == "__main__":
    main()