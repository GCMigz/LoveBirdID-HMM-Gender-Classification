"""
HMM-based Gender Classification for Fischer Lovebirds

This script implements a complete pipeline with:
1. Train/Test split (80/20) with stratified sampling
2. K-fold cross-validation on training set
3. Data augmentation applied per-fold to training data only
4. Comprehensive visualization system

Visualizations Generated (saved in visual_data/ folder):
- Individual sample visualizations (sample_XXX_original/augX_GENDER_complete.png):
  * Waveform
  * Spectrogram
  * MFCC and MFCC Delta
  * Mel Spectrogram
  * Fundamental Frequency (F0/Pitch)
  * Spectral Centroid
  * Zero Crossing Rate

- Gender Comparison (mfcc_gender_comparison.png):
  * Average male vs female MFCC patterns
  * Difference heatmap

- Cross-Validation Analysis (cv_results_analysis.png):
  * Confidence distribution by correctness
  * Confidence by gender
  * Accuracy progression
  * Prediction distribution

- Score Distributions (score_distributions.png):
  * Male vs Female model score scatter
  * Score difference distributions
  * Score boxplots by true gender

- Confusion Matrices:
  * CV and Test set confusion matrices

- Test Set Analysis (test_results_analysis.png):
  * Similar metrics for held-out test set

Model checkpoints saved every N samples for reproducibility.
"""

import os
import warnings
from dataclasses import dataclass

import joblib
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.max_open_warning'] = 100


@dataclass
class HMMConfig:
    sample_rate: int = 22050
    n_mfcc: int = 20
    n_fft: int = 1024
    hop_length: int = 256
    context_window: int = 2
    apply_cmvn: bool = True
    min_frames: int = 6
    max_frames_per_sequence: int = 2000
    n_iter: int = 60
    tol: float = 1e-3
    min_covar: float = 1e-3
    random_state: int = 42


@dataclass
class PipelineConfig:   #NOTE: Adjust paths as needed for your environment (WAGAN) 
    training_data_directory: str = "J:/Documents/Thesis/Code/audio/training_audio/masked"  # species root; expected layout: <species>/male and <species>/female
    output_directory: str = "J:/Documents/Thesis/Code/trained_model/masked"
    visual_data_directory: str = "J:/Documents/Thesis/Code/visual_data_Masked"
    genders: tuple = ("male", "female")
    cv_strategy: str = "stratified_kfold"  # options: stratified_kfold, leave_one_out
    cv_splits: int = 5  # Number of folds for K-fold (ignored for leave_one_out)
    n_jobs: int = 1  # use 1 to see detailed logs; -1 for parallel (faster but less verbose)
    show_sample_logs: bool = True
    use_data_augmentation: bool = True  # IMPORTANT: Augmentation applied ONLY to training data per fold
    checkpoint_every_n_samples: int = 90
    test_set_size: float = 0.2  # 20% held out for final test evaluation
    generate_visualizations: bool = False  # visual-data production disabled


FEATURE_CFG = HMMConfig()
PIPELINE_CFG = PipelineConfig()


def apply_cepstral_mean_variance_normalization(features, apply_variance_norm=True, epsilon=1e-8):
    mean = np.mean(features, axis=0)
    normalized_features = features - mean
    if apply_variance_norm:
        std_dev = np.std(normalized_features, axis=0)
        std_dev[std_dev < epsilon] = 1.0
        normalized_features = normalized_features / std_dev
    return normalized_features


def stack_neighboring_frames(features, left_context=2, right_context=2):
    if features is None or len(features) == 0:
        return features

    num_frames, num_features = features.shape
    total_context = left_context + 1 + right_context
    padded_features = np.pad(features, ((left_context, right_context), (0, 0)), mode="edge")

    stacked_features = np.zeros((num_frames, num_features * total_context), dtype=features.dtype)
    for index in range(num_frames):
        stacked_features[index] = padded_features[index:index + total_context].reshape(-1)

    return stacked_features


def limit_feature_frames(features, max_frames):
    if features is None or max_frames is None or max_frames <= 0:
        return features

    if len(features) <= max_frames:
        return features

    sample_indices = np.linspace(0, len(features) - 1, num=max_frames, dtype=int)
    return features[sample_indices]


def extract_mfcc_features_from_waveform(audio_waveform, cfg=FEATURE_CFG):
    mfcc_features = librosa.feature.mfcc(
        y=audio_waveform,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    mfcc_delta = librosa.feature.delta(mfcc_features)
    mfcc_delta_delta = librosa.feature.delta(mfcc_features, order=2)

    # Fundamental frequency (F0)
    f0 = librosa.yin(
        audio_waveform,
        fmin=50,
        fmax=5000,
        sr=cfg.sample_rate,
        frame_length=cfg.n_fft,
        hop_length=cfg.hop_length,
    )

    # Peak frequency from STFT peak bin per frame
    stft_magnitude = np.abs(librosa.stft(audio_waveform, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    peak_freq_bin = np.argmax(stft_magnitude, axis=0)
    peak_freq = peak_freq_bin * cfg.sample_rate / cfg.n_fft

    # Call duration is a clip-level value; repeat per frame as a static feature
    call_duration_seconds = len(audio_waveform) / max(cfg.sample_rate, 1)
    call_duration_vector = np.full_like(f0, fill_value=call_duration_seconds, dtype=np.float64)

    min_len = min(mfcc_features.shape[1], len(f0), len(peak_freq), len(call_duration_vector))
    mfcc_features = mfcc_features[:, :min_len]
    mfcc_delta = mfcc_delta[:, :min_len]
    mfcc_delta_delta = mfcc_delta_delta[:, :min_len]
    f0 = f0[:min_len]
    peak_freq = peak_freq[:min_len]
    call_duration_vector = call_duration_vector[:min_len]

    additional_features = np.vstack([f0, peak_freq, call_duration_vector])
    combined_features = np.vstack([mfcc_features, mfcc_delta, mfcc_delta_delta, additional_features]).T

    if cfg.apply_cmvn:
        combined_features = apply_cepstral_mean_variance_normalization(combined_features, apply_variance_norm=True)

    if cfg.context_window > 0:
        combined_features = stack_neighboring_frames(
            combined_features,
            left_context=cfg.context_window,
            right_context=cfg.context_window,
        )

    combined_features = limit_feature_frames(combined_features, cfg.max_frames_per_sequence)
    combined_features = np.asarray(combined_features, dtype=np.float32)

    return combined_features


def extract_mfcc_features_from_audio(audio_file_path, cfg=FEATURE_CFG):
    try:
        audio_signal, _ = librosa.load(audio_file_path, sr=cfg.sample_rate)
        return extract_mfcc_features_from_waveform(audio_signal, cfg)
    except Exception as error:
        print(f"Error processing {audio_file_path}: {error}")
        return None


def augment_audio(audio_waveform, sr):
    augmented_waveforms = [audio_waveform]

    # Slight pitch shift
    augmented_waveforms.append(librosa.effects.pitch_shift(audio_waveform, sr=sr, n_steps=1))

    # Slight time stretch
    augmented_waveforms.append(librosa.effects.time_stretch(audio_waveform, rate=0.9))

    # Add light gaussian noise
    noise = np.random.normal(0, 0.005, len(audio_waveform))
    augmented_waveforms.append(audio_waveform + noise)

    return augmented_waveforms


def visualize_sample(audio_path, sample_id, label, output_dir, augmented_idx=None, source_filename=None, augmentation_type=None):
    """Generate comprehensive visualizations for a single audio sample."""
    try:
        y, sr = librosa.load(audio_path, sr=FEATURE_CFG.sample_rate)

        os.makedirs(output_dir, exist_ok=True)

        base_name = source_filename if source_filename else os.path.splitext(os.path.basename(audio_path))[0]
        if augmented_idx is None:
            aug_tag = "original"
            aug_desc = "original"
        else:
            aug_tag = f"aug{augmented_idx}"
            default_aug_desc = {
                1: "pitch_shift_+1_semitone",
                2: "time_stretch_0.9x",
                3: "gaussian_noise_std0.005",
            }
            aug_desc = augmentation_type if augmentation_type else default_aug_desc.get(augmented_idx, f"augmentation_{augmented_idx}")

        prefix = f"{base_name}_{aug_tag}_{label}"
        figure_name = f"{base_name} | {label} | {aug_tag} ({aug_desc})"
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(figure_name, fontsize=14, fontweight='bold', y=0.995)
        
        # 1. Waveform
        plt.subplot(4, 2, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.8)
        plt.title(f'Waveform - {label.upper()}', fontsize=10)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # 2. Spectrogram
        plt.subplot(4, 2, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram', fontsize=10)
        
        # 3. MFCC
        plt.subplot(4, 2, 3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=FEATURE_CFG.n_mfcc)
        librosa.display.specshow(mfcc, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title('MFCC', fontsize=10)
        plt.ylabel('MFCC Coefficients')
        
        # 4. MFCC Delta
        plt.subplot(4, 2, 4)
        mfcc_delta = librosa.feature.delta(mfcc)
        librosa.display.specshow(mfcc_delta, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title('MFCC Delta', fontsize=10)
        plt.ylabel('MFCC Delta')
        
        # 5. Mel Spectrogram
        plt.subplot(4, 2, 5)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram', fontsize=10)
        
        # 6. Fundamental Frequency (F0)
        plt.subplot(4, 2, 6)
        f0 = librosa.yin(y, fmin=50, fmax=5000, sr=sr)
        times = librosa.times_like(f0, sr=sr)
        plt.plot(times, f0, linewidth=1.5, color='blue' if label == 'male' else 'red')
        plt.title('Fundamental Frequency (F0)', fontsize=10)
        plt.xlabel('Time (s)')
        plt.ylabel('Hz')
        plt.grid(True, alpha=0.3)
        
        # 7. Spectral Centroid
        plt.subplot(4, 2, 7)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sr)
        plt.plot(t, spectral_centroids, linewidth=1.5, color='blue' if label == 'male' else 'red')
        plt.title('Spectral Centroid', fontsize=10)
        plt.xlabel('Time (s)')
        plt.ylabel('Hz')
        plt.grid(True, alpha=0.3)
        
        # 8. Zero Crossing Rate
        plt.subplot(4, 2, 8)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        frames = range(len(zcr))
        t = librosa.frames_to_time(frames, sr=sr)
        plt.plot(t, zcr, linewidth=1.5, color='blue' if label == 'male' else 'red')
        plt.title('Zero Crossing Rate', fontsize=10)
        plt.xlabel('Time (s)')
        plt.ylabel('ZCR')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(output_dir, f"{prefix}_complete.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    except Exception as e:
        print(f"Error visualizing {audio_path}: {e}")
        return None


def visualize_mfcc_comparison(male_mfccs, female_mfccs, output_dir):
    """Compare average MFCC features between male and female samples."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("mfcc_gender_comparison", fontsize=14, fontweight='bold', y=1.02)
    
    # Average male MFCC
    avg_male = np.mean(male_mfccs, axis=0)
    im1 = axes[0].imshow(avg_male.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Average Male MFCC', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('MFCC Coefficients')
    plt.colorbar(im1, ax=axes[0])
    
    # Average female MFCC
    avg_female = np.mean(female_mfccs, axis=0)
    im2 = axes[1].imshow(avg_female.T, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Average Female MFCC', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('MFCC Coefficients')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff = avg_female - avg_male
    im3 = axes[2].imshow(diff.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axes[2].set_title('Difference (Female - Male)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time Frames')
    axes[2].set_ylabel('MFCC Coefficients')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(output_dir, "mfcc_gender_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def visualize_cv_results(detailed_results, output_dir, plot_name="cv_results_analysis"):
    """Visualize cross-validation predictions and confidence scores."""
    df = pd.DataFrame(detailed_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(plot_name, fontsize=14, fontweight='bold', y=0.995)
    
    # 1. Confidence distribution by correctness
    ax = axes[0, 0]
    correct_conf = df[df['correct'] == True]['confidence']
    incorrect_conf = df[df['correct'] == False]['confidence']
    ax.hist([correct_conf, incorrect_conf], bins=20, label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
    ax.set_xlabel('Confidence Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Confidence by true label
    ax = axes[0, 1]
    male_conf = df[df['true_label'] == 'male']['confidence']
    female_conf = df[df['true_label'] == 'female']['confidence']
    positions = [1, 2]
    bp = ax.boxplot([male_conf, female_conf], positions=positions, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax.set_xticks(positions)
    ax.set_xticklabels(['Male', 'Female'], fontsize=11)
    ax.set_ylabel('Confidence Score', fontsize=11)
    ax.set_title('Confidence Distribution by Gender', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Accuracy progression over folds
    ax = axes[1, 0]
    cumulative_correct = []
    cumulative_total = []
    running_correct = 0
    running_total = 0
    
    for i, result in enumerate(df.itertuples(), 1):
        if result.predicted_label != 'unknown':
            running_total += 1
            if result.correct:
                running_correct += 1
            if running_total > 0:
                cumulative_correct.append(running_correct / running_total)
                cumulative_total.append(i)
    
    ax.plot(cumulative_total, cumulative_correct, linewidth=2, color='blue', marker='o', markersize=2)
    ax.axhline(y=np.mean(cumulative_correct), color='red', linestyle='--', label=f'Mean: {np.mean(cumulative_correct):.3f}')
    ax.set_xlabel('Sample Number', fontsize=11)
    ax.set_ylabel('Cumulative Accuracy', fontsize=11)
    ax.set_title('Accuracy Progression Throughout CV', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 4. Prediction counts
    ax = axes[1, 1]
    pred_counts = df['predicted_label'].value_counts()
    colors = {'male': 'skyblue', 'female': 'pink', 'unknown': 'gray'}
    bars = ax.bar(pred_counts.index, pred_counts.values, color=[colors.get(x, 'gray') for x in pred_counts.index], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f"{plot_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def visualize_confusion_matrix(cm, labels, output_dir, title="Confusion Matrix"):
    """Create a detailed confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(8, 7))
    filename = title.lower().replace(" ", "_")
    fig.suptitle(filename, fontsize=14, fontweight='bold', y=0.99)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Count'})
    
    # Add normalized percentages
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]:.1%})',
                          ha="center", va="center", color="red", fontsize=9)
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, filename + ".png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def visualize_score_distributions(detailed_results, output_dir):
    """Visualize log-likelihood score distributions."""
    df = pd.DataFrame(detailed_results)
    
    # Extract scores
    male_scores = []
    female_scores = []
    true_labels = []
    
    for _, row in df.iterrows():
        if 'male' in row['log_likelihoods'] and 'female' in row['log_likelihoods']:
            male_scores.append(row['log_likelihoods']['male'])
            female_scores.append(row['log_likelihoods']['female'])
            true_labels.append(row['true_label'])
    
    male_scores = np.array(male_scores)
    female_scores = np.array(female_scores)
    true_labels = np.array(true_labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("score_distributions", fontsize=14, fontweight='bold', y=0.995)
    
    # 1. Score scatter plot
    ax = axes[0, 0]
    male_mask = true_labels == 'male'
    female_mask = true_labels == 'female'
    
    ax.scatter(male_scores[male_mask], female_scores[male_mask], 
              c='blue', label='True Male', alpha=0.6, s=50, edgecolors='black')
    ax.scatter(male_scores[female_mask], female_scores[female_mask], 
              c='red', label='True Female', alpha=0.6, s=50, edgecolors='black')
    
    # Add diagonal line
    min_val = min(male_scores.min(), female_scores.min())
    max_val = max(male_scores.max(), female_scores.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Male Model Score', fontsize=11)
    ax.set_ylabel('Female Model Score', fontsize=11)
    ax.set_title('Log-Likelihood Score Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Score difference distribution
    ax = axes[0, 1]
    score_diff = female_scores - male_scores
    male_diff = score_diff[male_mask]
    female_diff = score_diff[female_mask]
    
    ax.hist([male_diff, female_diff], bins=25, label=['True Male', 'True Female'], 
            alpha=0.7, color=['blue', 'red'])
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Score Difference (Female - Male)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Score Difference Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Male scores by true label
    ax = axes[1, 0]
    ax.boxplot([male_scores[male_mask], male_scores[female_mask]], 
               labels=['True Male', 'True Female'], patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax.set_ylabel('Male Model Score', fontsize=11)
    ax.set_title('Male Model Scores by True Gender', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Female scores by true label
    ax = axes[1, 1]
    ax.boxplot([female_scores[male_mask], female_scores[female_mask]], 
               labels=['True Male', 'True Female'], patch_artist=True,
               boxprops=dict(facecolor='lightpink', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax.set_ylabel('Female Model Score', fontsize=11)
    ax.set_title('Female Model Scores by True Gender', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "score_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def create_summary_report(cv_results, test_results, cv_accuracy, test_accuracy, output_dir):
    """Create a comprehensive summary visualization."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle('complete_summary_report', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    cv_df = pd.DataFrame(cv_results)
    test_df = pd.DataFrame(test_results)
    
    # 1. Overall Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [cv_accuracy, test_accuracy]
    bars = ax1.bar(['CV (Training)', 'Test (Hold-out)'], accuracies, 
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Sample Counts
    ax2 = fig.add_subplot(gs[0, 1])
    sample_counts = {
        'CV Train': len(cv_df),
        'Test': len(test_df)
    }
    bars = ax2.bar(sample_counts.keys(), sample_counts.values(), 
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax2.set_title('Dataset Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Confidence Statistics
    ax3 = fig.add_subplot(gs[0, 2])
    cv_conf = cv_df['confidence']
    test_conf = test_df['confidence']
    bp = ax3.boxplot([cv_conf, test_conf], labels=['CV', 'Test'], 
                     patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Confidence Score', fontsize=11, fontweight='bold')
    ax3.set_title('Confidence Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. CV Gender Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    cv_gender_true = cv_df['true_label'].value_counts()
    ax4.pie(cv_gender_true.values, labels=[l.upper() for l in cv_gender_true.index], 
           autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90,
           textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax4.set_title('CV Set - True Gender Distribution', fontsize=11, fontweight='bold')
    
    # 5. Test Gender Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    test_gender_true = test_df['true_label'].value_counts()
    ax5.pie(test_gender_true.values, labels=[l.upper() for l in test_gender_true.index], 
           autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90,
           textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax5.set_title('Test Set - True Gender Distribution', fontsize=11, fontweight='bold')
    
    # 6. Correctness Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    cv_correct = cv_df['correct'].value_counts()
    test_correct = test_df['correct'].value_counts()
    x = np.arange(2)
    width = 0.35
    cv_vals = [cv_correct.get(True, 0), cv_correct.get(False, 0)]
    test_vals = [test_correct.get(True, 0), test_correct.get(False, 0)]
    ax6.bar(x - width/2, cv_vals, width, label='CV', color='#3498db', alpha=0.7, edgecolor='black')
    ax6.bar(x + width/2, test_vals, width, label='Test', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax6.set_title('Correct vs Incorrect Predictions', fontsize=11, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Correct', 'Incorrect'], fontsize=10)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. CV Confidence by Correctness
    ax7 = fig.add_subplot(gs[2, 0])
    cv_correct_conf = cv_df[cv_df['correct'] == True]['confidence']
    cv_incorrect_conf = cv_df[cv_df['correct'] == False]['confidence']
    ax7.hist([cv_correct_conf, cv_incorrect_conf], bins=15, 
            label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
    ax7.set_xlabel('Confidence', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title('CV: Confidence Distribution', fontsize=11, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Test Confidence by Correctness
    ax8 = fig.add_subplot(gs[2, 1])
    test_correct_conf = test_df[test_df['correct'] == True]['confidence']
    test_incorrect_conf = test_df[test_df['correct'] == False]['confidence']
    ax8.hist([test_correct_conf, test_incorrect_conf], bins=15, 
            label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
    ax8.set_xlabel('Confidence', fontsize=10)
    ax8.set_ylabel('Frequency', fontsize=10)
    ax8.set_title('Test: Confidence Distribution', fontsize=11, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Key Metrics Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    cv_avg_conf = cv_df['confidence'].mean()
    test_avg_conf = test_df['confidence'].mean()
    cv_male_acc = (cv_df[(cv_df['true_label'] == 'male') & (cv_df['correct'] == True)].shape[0] / 
                   cv_df[cv_df['true_label'] == 'male'].shape[0]) if len(cv_df[cv_df['true_label'] == 'male']) > 0 else 0
    cv_female_acc = (cv_df[(cv_df['true_label'] == 'female') & (cv_df['correct'] == True)].shape[0] / 
                     cv_df[cv_df['true_label'] == 'female'].shape[0]) if len(cv_df[cv_df['true_label'] == 'female']) > 0 else 0
    test_male_acc = (test_df[(test_df['true_label'] == 'male') & (test_df['correct'] == True)].shape[0] / 
                     test_df[test_df['true_label'] == 'male'].shape[0]) if len(test_df[test_df['true_label'] == 'male']) > 0 else 0
    test_female_acc = (test_df[(test_df['true_label'] == 'female') & (test_df['correct'] == True)].shape[0] / 
                       test_df[test_df['true_label'] == 'female'].shape[0]) if len(test_df[test_df['true_label'] == 'female']) > 0 else 0
    
    table_data = [
        ['Metric', 'CV', 'Test'],
        ['Overall Acc.', f'{cv_accuracy:.3f}', f'{test_accuracy:.3f}'],
        ['Avg Confidence', f'{cv_avg_conf:.3f}', f'{test_avg_conf:.3f}'],
        ['Male Acc.', f'{cv_male_acc:.3f}', f'{test_male_acc:.3f}'],
        ['Female Acc.', f'{cv_female_acc:.3f}', f'{test_female_acc:.3f}'],
    ]
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax9.set_title('Performance Metrics Summary', fontsize=11, fontweight='bold', pad=20)
    
    save_path = os.path.join(output_dir, "complete_summary_report.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def _save_checkpoint_model(model, gender, sample_count, output_directory):
    checkpoint_path = os.path.join(output_directory, f"{gender}HMM_{sample_count}samples.pkl")
    joblib.dump(model, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def train_gender_hmm_model(feature_sequences, num_hidden_states=4, cfg=FEATURE_CFG):
    valid_sequences = [sequence for sequence in feature_sequences if sequence is not None and len(sequence) >= cfg.min_frames]
    if not valid_sequences:
        return None

    valid_sequences = [np.asarray(sequence, dtype=np.float32) for sequence in valid_sequences]
    sequence_lengths = [len(sequence) for sequence in valid_sequences]
    training_matrix = np.vstack(valid_sequences).astype(np.float32, copy=False)

    hmm_model = hmm.GaussianHMM(
        n_components=num_hidden_states,
        covariance_type="diag",
        n_iter=cfg.n_iter,
        tol=cfg.tol,
        random_state=cfg.random_state,
        min_covar=cfg.min_covar,
    )
    hmm_model.fit(training_matrix, lengths=sequence_lengths)
    return hmm_model


def calculate_prediction_confidence_from_loglikelihood(gender_loglikelihood_scores, temperature=8.0, max_confidence=0.95):
    if not gender_loglikelihood_scores:
        return 0.5

    try:
        values = np.array(list(gender_loglikelihood_scores.values()), dtype=np.float64)
        values = values - np.max(values)
        probabilities = np.exp(values / max(temperature, 1e-6))
        probabilities = probabilities / (np.sum(probabilities) + 1e-12)
        confidence = float(np.max(probabilities))
        return min(max(confidence, 0.1), max_confidence)
    except Exception:
        return 0.5


def _scan_audio_files(training_data_directory, genders):
    all_paths = []
    all_labels = []

    valid_genders = {gender.lower() for gender in genders}
    if not os.path.exists(training_data_directory):
        return all_paths, all_labels

    # Expected layout: audio/training_audio/<species>/male|female
    for gender in sorted(valid_genders):
        gender_directory = os.path.join(training_data_directory, gender)
        if not os.path.isdir(gender_directory):
            continue

        for filename in sorted(os.listdir(gender_directory)):
            if filename.lower().endswith((".wav", ".mp3", ".flac")):
                all_paths.append(os.path.join(gender_directory, filename))
                all_labels.append(gender)

    return all_paths, all_labels


def _print_dataset_breakdown(paths, labels):
    """Print discovered dataset composition for the configured species folder."""
    print("\n" + "=" * 70)
    print("DATA DISCOVERY SUMMARY")
    print("=" * 70)

    gender_counts = {"male": 0, "female": 0}
    species_gender_counts = {}
    configured_species = os.path.basename(os.path.normpath(PIPELINE_CFG.training_data_directory))

    for path, label in zip(paths, labels):
        gender = str(label).lower()
        if gender not in gender_counts:
            gender_counts[gender] = 0
        gender_counts[gender] += 1

        species = configured_species or "root"

        if species not in species_gender_counts:
            species_gender_counts[species] = {"male": 0, "female": 0}
        if gender not in species_gender_counts[species]:
            species_gender_counts[species][gender] = 0
        species_gender_counts[species][gender] += 1

    print(f"Total discovered files: {len(paths)}")
    print(f"  Male: {gender_counts.get('male', 0)}")
    print(f"  Female: {gender_counts.get('female', 0)}")
    print("By species root:")
    for species in sorted(species_gender_counts):
        male_count = species_gender_counts[species].get("male", 0)
        female_count = species_gender_counts[species].get("female", 0)
        total_count = male_count + female_count
        print(f"  - {species}: total={total_count}, male={male_count}, female={female_count}")
    print("=" * 70)


def _adaptive_num_states(labels):
    male_count = sum(1 for label in labels if label == "male")
    female_count = sum(1 for label in labels if label == "female")
    samples_per_class = min(male_count, female_count)

    if samples_per_class < 15:
        return 3
    if samples_per_class < 40:
        return 4
    return 5


def _safe_avg_loglikelihood(model, features):
    try:
        return float(model.score(features) / max(len(features), 1))
    except Exception:
        return float("-inf")


def _build_cv_splits(paths, labels):
    if PIPELINE_CFG.cv_strategy == "stratified_kfold":
        male_count = sum(1 for label in labels if label == "male")
        female_count = sum(1 for label in labels if label == "female")
        max_valid_splits = min(male_count, female_count)
        n_splits = min(PIPELINE_CFG.cv_splits, max_valid_splits)

        if n_splits >= 2:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=FEATURE_CFG.random_state,
            )
            return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(paths, labels)]

    splitter = LeaveOneOut()
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(paths)]


def _evaluate_single_fold(train_indices, test_indices, all_audio_file_paths, ground_truth_gender_labels, feature_cache, num_hmm_hidden_states, fold_info=""):
    fold_label = fold_info if fold_info else "FOLD"
    fallback_label = "unknown"
    if PIPELINE_CFG.show_sample_logs and fold_info:
        print("\n" + "=" * 70)
        print(fold_info)
        print("=" * 70)

    # CRITICAL: Augment training data only (not test data)
    trained_models = {}
    for gender in PIPELINE_CFG.genders:
        class_train_features = []
        for index in train_indices:
            if ground_truth_gender_labels[index] == gender:
                original_features = feature_cache[index]
                class_train_features.append(original_features)
                
                # Apply augmentation only to training data
                if PIPELINE_CFG.use_data_augmentation:
                    # Load original audio for augmentation
                    try:
                        audio_signal, _ = librosa.load(all_audio_file_paths[index], sr=FEATURE_CFG.sample_rate)
                        augmented_waveforms = augment_audio(audio_signal, FEATURE_CFG.sample_rate)
                        # Skip first waveform (original, already added)
                        for aug_waveform in augmented_waveforms[1:]:
                            aug_features = extract_mfcc_features_from_waveform(aug_waveform, FEATURE_CFG)
                            if aug_features is not None and len(aug_features) >= FEATURE_CFG.min_frames:
                                class_train_features.append(aug_features)
                    except Exception as e:
                        pass  # Skip augmentation if error occurs
        
        if len(class_train_features) >= 2:
            model = train_gender_hmm_model(class_train_features, num_hmm_hidden_states)
            if model is not None:
                trained_models[gender] = model

    fold_results = []
    total_fold_tests = len(test_indices)

    for sample_idx, test_index in enumerate(test_indices, start=1):
        test_index = int(test_index)
        test_file = all_audio_file_paths[test_index]
        true_label = ground_truth_gender_labels[test_index]
        test_features = feature_cache[test_index]  # Original, non-augmented test features

        if PIPELINE_CFG.show_sample_logs:
            print("\n" + "-" * 70)
            print(f"{fold_label} | SAMPLE {sample_idx}/{total_fold_tests}")
            print(f"File: {os.path.basename(test_file)}")
            print(f"GROUND TRUTH (TRUE GENDER): {true_label.upper()}")

        if not trained_models:
            fold_results.append(
                {
                    "filename": os.path.basename(test_file),
                    "true_label": true_label,
                    "raw_predicted_label": fallback_label,
                    "predicted_label": fallback_label,
                    "log_likelihoods": {},
                    "confidence": 0.0,
                    "diff": 0.0,
                    "correct": fallback_label == true_label,
                }
            )
            continue

        scores = {}
        for gender, model in trained_models.items():
            score = _safe_avg_loglikelihood(model, test_features)
            if np.isfinite(score):
                scores[gender] = score

        if not scores:
            fold_results.append(
                {
                    "filename": os.path.basename(test_file),
                    "true_label": true_label,
                    "raw_predicted_label": fallback_label,
                    "predicted_label": fallback_label,
                    "log_likelihoods": {},
                    "confidence": 0.0,
                    "diff": 0.0,
                    "correct": fallback_label == true_label,
                }
            )
            continue

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        raw_predicted_label = sorted_scores[0][0]
        score_diff = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        confidence = calculate_prediction_confidence_from_loglikelihood(scores)
        predicted_label = raw_predicted_label

        if PIPELINE_CFG.show_sample_logs:
            print(f"Scores: Male={scores.get('male', 'N/A'):.4f}, Female={scores.get('female', 'N/A'):.4f}")
            print(f"PREDICTED: {predicted_label.upper()} (confidence: {confidence:.3f})")
            print(f"Result: {' CORRECT' if predicted_label == true_label else ' INCORRECT'}")

        fold_results.append(
            {
                "filename": os.path.basename(test_file),
                "true_label": true_label,
                "raw_predicted_label": raw_predicted_label,
                "predicted_label": predicted_label,
                "log_likelihoods": scores,
                "confidence": confidence,
                "diff": float(score_diff),
                "correct": predicted_label == true_label,
            }
        )

    return fold_results


def _save_outputs(output_dir, classification_accuracy, classification_report_text, confusion_matrix_result, detailed_results, metadata_note=""):
    os.makedirs(output_dir, exist_ok=True)

    cv_name = f"{PIPELINE_CFG.cv_splits}-Fold Stratified CV" if PIPELINE_CFG.cv_strategy == "stratified_kfold" else "Leave-One-Out CV"

    with open(os.path.join(output_dir, "hmm_model_evaluation.txt"), "w", encoding="utf-8") as evaluation_file:
        evaluation_file.write(f"HMM Model Evaluation ({cv_name})\n")
        evaluation_file.write("=" * 50 + "\n")
        if metadata_note:
            evaluation_file.write(metadata_note)
            evaluation_file.write("=" * 50 + "\n")
        evaluation_file.write(f"Total audio files: {len(detailed_results)}\n")
        evaluation_file.write(f"Predictions: {len(detailed_results)}\n")
        evaluation_file.write(f"Accuracy: {classification_accuracy:.4f}\n\n")
        evaluation_file.write("Classification Report:\n")
        evaluation_file.write(classification_report_text + "\n")
        evaluation_file.write("Confusion Matrix:\n")
        evaluation_file.write(str(confusion_matrix_result) + "\n\n")
        evaluation_file.write("Detailed Results:\n")
        for result in detailed_results:
            evaluation_file.write(
                f"{result['filename']}: True={result['true_label']}, Pred={result['predicted_label']}, "
                f"Conf={result['confidence']:.3f}, Correct={result['correct']}\n"
            )

    csv_rows = []
    for result in detailed_results:
        male_score = result["log_likelihoods"].get("male")
        female_score = result["log_likelihoods"].get("female")

        csv_rows.append(
            {
                "filename": result["filename"],
                "true_label": result["true_label"],
                "predicted_label": result["predicted_label"],
                "confidence": f"{result['confidence']:.4f}",
                "correct": result["correct"],
                "male_score": "N/A" if male_score is None or np.isneginf(male_score) else f"{male_score:.6f}",
                "female_score": "N/A" if female_score is None or np.isneginf(female_score) else f"{female_score:.6f}",
                "score_difference": f"{result['diff']:.6f}",
            }
        )

    pd.DataFrame(csv_rows).to_csv(os.path.join(output_dir, "detailed_predictions.csv"), index=False)

    with open(os.path.join(output_dir, "summary_metrics.txt"), "w", encoding="utf-8") as metrics_file:
        metrics_file.write("FISCHER LOVEBIRD HMM MODEL - SUMMARY METRICS\n")
        metrics_file.write("=" * 60 + "\n\n")
        metrics_file.write("OVERALL PERFORMANCE:\n")
        metrics_file.write(f"  Accuracy:              {classification_accuracy:.4f}\n")
        metrics_file.write(f"  Total Test Samples:    {len(detailed_results)}\n")
        metrics_file.write(f"  Predictions:           {len(detailed_results)}\n")
        metrics_file.write(f"  Average Confidence:    {np.mean([result['confidence'] for result in detailed_results]):.4f}\n\n")
        metrics_file.write("PER-CLASS METRICS (from Classification Report):\n")
        metrics_file.write(classification_report_text + "\n\n")
        metrics_file.write("CONFUSION MATRIX:\n")
        metrics_file.write("(Rows=True Label, Columns=Predicted Label)\n")
        metrics_file.write(str(confusion_matrix_result) + "\n")


def train_gender_classification_hmm_models():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    all_audio_file_paths, ground_truth_gender_labels = _scan_audio_files(
        PIPELINE_CFG.training_data_directory,
        PIPELINE_CFG.genders,
    )
    if not all_audio_file_paths:
        print("No audio files found! Please check your directory structure.")
        return None

    _print_dataset_breakdown(all_audio_file_paths, ground_truth_gender_labels)

    # Split into train/test BEFORE any processing
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_audio_file_paths,
        ground_truth_gender_labels,
        test_size=PIPELINE_CFG.test_set_size,
        random_state=FEATURE_CFG.random_state,
        stratify=ground_truth_gender_labels
    )

    num_hmm_hidden_states = _adaptive_num_states(train_labels)

    print("=" * 70)
    print(f"DATASET SPLIT: {len(all_audio_file_paths)} total samples")
    print(f"  Training: {len(train_paths)} samples ({len([l for l in train_labels if l=='male'])} male, {len([l for l in train_labels if l=='female'])} female)")
    print(f"  Test (held-out): {len(test_paths)} samples ({len([l for l in test_labels if l=='male'])} male, {len([l for l in test_labels if l=='female'])} female)")
    print(f"Data augmentation: {'ON (applied per-fold to training data only)' if PIPELINE_CFG.use_data_augmentation else 'OFF'}")
    print("=" * 70)

    # Extract features from TRAINING samples only (no augmentation yet)
    print("\nExtracting features from training set...")
    train_feature_sequences = []
    valid_train_paths = []
    valid_train_labels = []

    for index, audio_path in enumerate(train_paths):
        label = train_labels[index]
        try:
            audio_signal, _ = librosa.load(audio_path, sr=FEATURE_CFG.sample_rate)
            features = extract_mfcc_features_from_waveform(audio_signal, FEATURE_CFG)
            
            if features is not None and len(features) >= FEATURE_CFG.min_frames:
                train_feature_sequences.append(features)
                valid_train_paths.append(audio_path)
                valid_train_labels.append(label)
        except Exception as error:
            print(f"Error loading {audio_path}: {error}")
            continue

    if len(train_feature_sequences) < 6:
        print("Not enough valid feature sequences for training.")
        return None

    # Extract features from TEST samples
    print("Extracting features from test set...")
    test_feature_sequences = []
    valid_test_paths = []
    valid_test_labels = []

    for index, audio_path in enumerate(test_paths):
        label = test_labels[index]
        try:
            audio_signal, _ = librosa.load(audio_path, sr=FEATURE_CFG.sample_rate)
            features = extract_mfcc_features_from_waveform(audio_signal, FEATURE_CFG)
            
            if features is not None and len(features) >= FEATURE_CFG.min_frames:
                test_feature_sequences.append(features)
                valid_test_paths.append(audio_path)
                valid_test_labels.append(label)
        except Exception as error:
            print(f"Error loading test {audio_path}: {error}")
            continue

    all_audio_file_paths = valid_train_paths
    ground_truth_gender_labels = valid_train_labels
    feature_cache = {index: sequence for index, sequence in enumerate(train_feature_sequences)}
    
    test_feature_cache = {index: sequence for index, sequence in enumerate(test_feature_sequences)}

    # ========================================================================
    # GENERATE VISUALIZATIONS FOR ALL SAMPLES
    # ========================================================================
    # NOTE: Visualization generation commented out - can be re-enabled later
    # if PIPELINE_CFG.generate_visualizations:
    #     print("\n" + "=" * 70)
    #     print("GENERATING VISUALIZATIONS")
    #     print("=" * 70)
    #     
    #     vis_dir = PIPELINE_CFG.visual_data_directory
    #     os.makedirs(vis_dir, exist_ok=True)
    #     
    #     # Visualize all training samples (original + augmented)
    #     print("Visualizing training samples...")
    #     sample_counter = 1
    #     male_mfccs = []
    #     female_mfccs = []
    #     
    #     for idx, (path, label) in enumerate(zip(valid_train_paths, valid_train_labels)):
    #         # Visualize original sample
    #         source_stem = os.path.splitext(os.path.basename(path))[0]
    #         visualize_sample(path, sample_counter, label, vis_dir, augmented_idx=None, source_filename=source_stem)
    #         
    #         # Store MFCC for comparison
    #         if label == 'male':
    #             male_mfccs.append(feature_cache[idx])
    #         else:
    #             female_mfccs.append(feature_cache[idx])
    #         
    #         # Visualize augmented versions if augmentation is enabled
    #         if PIPELINE_CFG.use_data_augmentation:
    #             try:
    #                 audio_signal, _ = librosa.load(path, sr=FEATURE_CFG.sample_rate)
    #                 augmented_waveforms = augment_audio(audio_signal, FEATURE_CFG.sample_rate)
    #                 
    #                 for aug_idx, aug_waveform in enumerate(augmented_waveforms[1:], start=1):
    #                     # Save augmented audio temporarily for visualization
    #                     temp_aug_path = os.path.join(vis_dir, f"_temp_aug_{sample_counter}_{aug_idx}.wav")
    #                     import soundfile as sf
    #                     sf.write(temp_aug_path, aug_waveform, FEATURE_CFG.sample_rate)
    #                     
    #                     augmentation_map = {
    #                         1: "pitch_shift_+1_semitone",
    #                         2: "time_stretch_0.9x",
    #                         3: "gaussian_noise_std0.005",
    #                     }
    #                     visualize_sample(
    #                         temp_aug_path,
    #                         sample_counter,
    #                         label,
    #                         vis_dir,
    #                         augmented_idx=aug_idx,
    #                         source_filename=source_stem,
    #                         augmentation_type=augmentation_map.get(aug_idx),
    #                     )
    #                     
    #                     # Clean up temp file
    #                     if os.path.exists(temp_aug_path):
    #                         os.remove(temp_aug_path)
    #             except Exception as e:
    #                 print(f"Error visualizing augmented versions of sample {sample_counter}: {e}")
    #         
    #         sample_counter += 1
    #         
    #         if idx % 10 == 0:
    #             print(f"  Progress: {idx}/{len(valid_train_paths)} training samples processed")
    #     
    #     # Visualize test samples
    #     print("Visualizing test samples...")
    #     for idx, (path, label) in enumerate(zip(valid_test_paths, valid_test_labels)):
    #         visualize_sample(path, sample_counter, label, vis_dir, augmented_idx=None)
    #         sample_counter += 1
    #         
    #     # Create gender comparison visualization
    #     print("Creating MFCC gender comparison...")
    #     if male_mfccs and female_mfccs:
    #         visualize_mfcc_comparison(male_mfccs, female_mfccs, vis_dir)
    #     
    #     print(f"Visualizations saved to: {vis_dir}")
    #     print("=" * 70)

    print(f"Valid ORIGINAL samples: {len(all_audio_file_paths)}")
    if PIPELINE_CFG.use_data_augmentation:
        print(f"Training samples per fold (with augmentation): ~{len(all_audio_file_paths) * 4 * (PIPELINE_CFG.cv_splits - 1) // PIPELINE_CFG.cv_splits}")
    print(f"Using HMM states: {num_hmm_hidden_states}")
    print(f"HMM max iterations per fit: {FEATURE_CFG.n_iter}")
    print("Prediction mode: forced binary (male/female), no unknown class")

    cv_splits = _build_cv_splits(all_audio_file_paths, ground_truth_gender_labels)
    detailed_results = []

    cv_name = f"{len(cv_splits)}-fold Stratified K-Fold" if PIPELINE_CFG.cv_strategy == "stratified_kfold" else "Leave-One-Out"
    print("\n" + "=" * 70)
    print(f"STARTING CROSS-VALIDATION: {cv_name} ({len(cv_splits)} folds)")
    print(f"Total ORIGINAL test samples: {len(all_audio_file_paths)}")
    print("=" * 70)

    if PIPELINE_CFG.n_jobs == 1:
        for fold_index, (train_indices, test_indices) in enumerate(cv_splits, start=1):
            fold_info = f"FOLD {fold_index}/{len(cv_splits)} | Test samples: {len(test_indices)}"
            detailed_results.extend(
                _evaluate_single_fold(
                    train_indices,
                    test_indices,
                    all_audio_file_paths,
                    ground_truth_gender_labels,
                    feature_cache,
                    num_hmm_hidden_states,
                    fold_info=fold_info,
                )
            )
    else:
        print("Running in parallel mode (no per-sample logs)...")
        fold_results = joblib.Parallel(n_jobs=PIPELINE_CFG.n_jobs, prefer="threads", verbose=10)(
            joblib.delayed(_evaluate_single_fold)(
                train_indices,
                test_indices,
                all_audio_file_paths,
                ground_truth_gender_labels,
                feature_cache,
                num_hmm_hidden_states,
                fold_info="",
            )
            for train_indices, test_indices in cv_splits
        )
        detailed_results = [result for fold in fold_results for result in fold]

    if detailed_results:
        final_predicted_labels = [result["predicted_label"] for result in detailed_results]
        final_actual_labels = [result["true_label"] for result in detailed_results]
        classification_accuracy = accuracy_score(final_actual_labels, final_predicted_labels)
        classification_report_text = classification_report(
            final_actual_labels,
            final_predicted_labels,
            labels=list(PIPELINE_CFG.genders),
            target_names=list(PIPELINE_CFG.genders),
            zero_division=0,
        )
        confusion_matrix_result = confusion_matrix(
            final_actual_labels,
            final_predicted_labels,
            labels=list(PIPELINE_CFG.genders),
        )
    else:
        final_predicted_labels = []
        final_actual_labels = []
        classification_accuracy = 0.0
        classification_report_text = "No valid predictions"
        confusion_matrix_result = np.array([])

    print("\n" + "=" * 50)
    print(f"HMM MODEL EVALUATION ({cv_name})")
    print("=" * 50)
    print(f"Total files: {len(all_audio_file_paths)}")
    print(f"Valid predictions: {len(final_predicted_labels)}")
    print(f"Accuracy: {classification_accuracy:.4f}")

    os.makedirs(PIPELINE_CFG.output_directory, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("TRAINING FINAL MODELS ON ALL DATA")
    print("=" * 50)
    
    final_models = {}
    for gender in PIPELINE_CFG.genders:
        # Collect all features for this gender, with augmentation
        class_features = []
        for index, label in enumerate(ground_truth_gender_labels):
            if label == gender:
                # Add original features
                class_features.append(feature_cache[index])
                
                # Add augmented features if enabled
                if PIPELINE_CFG.use_data_augmentation:
                    try:
                        audio_signal, _ = librosa.load(all_audio_file_paths[index], sr=FEATURE_CFG.sample_rate)
                        augmented_waveforms = augment_audio(audio_signal, FEATURE_CFG.sample_rate)
                        # Skip first (original), add augmented versions
                        for aug_waveform in augmented_waveforms[1:]:
                            aug_features = extract_mfcc_features_from_waveform(aug_waveform, FEATURE_CFG)
                            if aug_features is not None and len(aug_features) >= FEATURE_CFG.min_frames:
                                class_features.append(aug_features)
                    except Exception:
                        pass
        
        if len(class_features) < 2:
            continue

        print(f"Training {gender} model with {len(class_features)} samples...")
        
        # Save checkpoints during training
        checkpoint_step = max(1, PIPELINE_CFG.checkpoint_every_n_samples)
        for checkpoint_size in range(checkpoint_step, len(class_features) + 1, checkpoint_step):
            checkpoint_model = train_gender_hmm_model(class_features[:checkpoint_size], num_hmm_hidden_states)
            if checkpoint_model is not None:
                _save_checkpoint_model(
                    checkpoint_model,
                    gender,
                    checkpoint_size,
                    PIPELINE_CFG.output_directory,
                )
        
        # Train final model on all data
        model = train_gender_hmm_model(class_features, num_hmm_hidden_states)
        if model is None:
            continue

        final_models[gender] = model
        model_path = os.path.join(PIPELINE_CFG.output_directory, f"{gender}HMM.pkl")
        joblib.dump(model, model_path)
        print(f"Saved final model: {model_path}")

    if final_models:
        combined_model_path = os.path.join(PIPELINE_CFG.output_directory, "gender_hmm_models.pkl")
        joblib.dump(
            {
                "models": final_models,
                "feature_config": FEATURE_CFG,
                "pipeline_config": PIPELINE_CFG,
            },
            combined_model_path,
        )
        print(f"Saved combined model bundle: {combined_model_path}")

    # Add metadata about methodology
    metadata_note = f"\\nMETHODOLOGY:\\n"
    metadata_note += f"1. Initial split: {int((1-PIPELINE_CFG.test_set_size)*100)}% train, {int(PIPELINE_CFG.test_set_size*100)}% test (stratified)\\n"
    metadata_note += f"2. Cross-validation performed ONLY on training set\\n"
    metadata_note += f"3. Data augmentation: {'ENABLED' if PIPELINE_CFG.use_data_augmentation else 'DISABLED'}\\n"
    if PIPELINE_CFG.use_data_augmentation:
        metadata_note += f"   - Applied ONLY to training data in each CV fold\\n"
        metadata_note += f"   - Test data remained original (non-augmented)\\n"
    metadata_note += f"4. Final model trained on all training data (with augmentation)\\n"
    metadata_note += f"5. Separate held-out test set evaluation saved in test_set_evaluation.txt\\n"
    
    _save_outputs(
        PIPELINE_CFG.output_directory,
        classification_accuracy,
        classification_report_text,
        confusion_matrix_result,
        detailed_results,
        metadata_note,
    )

    # ========================================================================
    # FINAL EVALUATION ON HELD-OUT TEST SET
    # ========================================================================
    print("\n" + "=" * 70)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("=" * 70)
    print(f"Test samples: {len(valid_test_paths)}")
    
    test_results = []
    for idx, test_path in enumerate(valid_test_paths):
        test_features = test_feature_cache[idx]
        true_label = valid_test_labels[idx]
        filename = os.path.basename(test_path)
        
        # Get predictions from final models
        scores = {}
        for gender, model in final_models.items():
            score = _safe_avg_loglikelihood(model, test_features)
            if np.isfinite(score):
                scores[gender] = score
        
        if not scores:
            predicted_label = "unknown"
            confidence = 0.0
            score_diff = 0.0
            print(f"Warning: no usable model scores for {filename}; marking as UNKNOWN.")
        else:
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            raw_predicted_label = sorted_scores[0][0]
            score_diff = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
            confidence = calculate_prediction_confidence_from_loglikelihood(scores)
            predicted_label = raw_predicted_label
        
        is_correct = predicted_label == true_label
        
        test_results.append({
            "filename": filename,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": is_correct,
            "male_score": scores.get("male", float("-inf")),
            "female_score": scores.get("female", float("-inf")),
            "score_difference": score_diff
        })
        
        # Log individual predictions
        print(f"\n{idx+1}. {filename}")
        print(f"   True: {true_label.upper()} | Predicted: {predicted_label.upper()} | Confidence: {confidence:.3f}")
        print(f"   Scores - Male: {scores.get('male', 'N/A'):.4f}, Female: {scores.get('female', 'N/A'):.4f}")
        print(f"   {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
    # Calculate test set metrics
    if test_results:
        test_pred = [r["predicted_label"] for r in test_results]
        test_true = [r["true_label"] for r in test_results]
        test_accuracy = accuracy_score(test_true, test_pred)
        test_report = classification_report(
            test_true,
            test_pred,
            labels=list(PIPELINE_CFG.genders),
            target_names=list(PIPELINE_CFG.genders),
            zero_division=0
        )
        test_cm = confusion_matrix(test_true, test_pred, labels=list(PIPELINE_CFG.genders))
    else:
        test_accuracy = 0.0
        test_report = "No valid predictions"
        test_cm = np.array([])
    
    # Save test results
    test_output_file = os.path.join(PIPELINE_CFG.output_directory, "test_set_evaluation.txt")
    with open(test_output_file, "w", encoding="utf-8") as f:
        f.write("HELD-OUT TEST SET EVALUATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test samples: {len(valid_test_paths)}\n")
        f.write(f"Predictions: {len(test_results)}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(test_report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(test_cm) + "\n\n")
        f.write("Detailed Predictions:\n")
        f.write("=" * 70 + "\n")
        for r in test_results:
            male_str = f"{r['male_score']:.6f}" if np.isfinite(r['male_score']) else "N/A"
            female_str = f"{r['female_score']:.6f}" if np.isfinite(r['female_score']) else "N/A"
            f.write(f"\n{r['filename']}\n")
            f.write(f"  True: {r['true_label']} | Predicted: {r['predicted_label']} | Confidence: {r['confidence']:.4f}\n")
            f.write(f"  Male Score: {male_str} | Female Score: {female_str}\n")
            f.write(f"  Result: {'CORRECT' if r['correct'] else 'INCORRECT'}\n")
    
    # Save test results CSV
    test_csv = os.path.join(PIPELINE_CFG.output_directory, "test_set_predictions.csv")
    pd.DataFrame(test_results).to_csv(test_csv, index=False)
    
    # ========================================================================
    # GENERATE ANALYSIS VISUALIZATIONS
    # ========================================================================
    if PIPELINE_CFG.generate_visualizations:
        print("\n" + "=" * 70)
        print("GENERATING ANALYSIS VISUALIZATIONS")
        print("=" * 70)
        
        vis_dir = PIPELINE_CFG.visual_data_directory
        
        # CV Results visualizations
        print("Creating CV results visualization...")
        visualize_cv_results(detailed_results, vis_dir, plot_name="cv_results_analysis")
        
        # Score distributions
        print("Creating score distribution plots...")
        visualize_score_distributions(detailed_results, vis_dir)
        
        # Confusion matrices
        print("Creating confusion matrices...")
        if len(detailed_results) > 0:
            visualize_confusion_matrix(confusion_matrix_result, 
                                      list(PIPELINE_CFG.genders), 
                                      vis_dir, 
                                      "CV Confusion Matrix")
        
        if len(test_results) > 0:
            visualize_confusion_matrix(test_cm, 
                                      list(PIPELINE_CFG.genders), 
                                      vis_dir, 
                                      "Test Set Confusion Matrix")
        
        # Test results visualization
        print("Creating test set analysis...")
        visualize_cv_results(test_results, vis_dir, plot_name="test_results_analysis")
        
        # Create comprehensive summary report
        print("Creating comprehensive summary report...")
        create_summary_report(detailed_results, test_results, 
                            classification_accuracy, test_accuracy, vis_dir)
        
        print(f"\nAll visualizations saved to: {vis_dir}")
        print("Generated files:")
        print("  - sample_XXX_original/augX_GENDER_complete.png (individual samples)")
        print("  - mfcc_gender_comparison.png")
        print("  - cv_results_analysis.png")
        print("  - test_results_analysis.png")
        print("  - score_distributions.png")
        print("  - cv_confusion_matrix.png")
        print("  - test_set_confusion_matrix.png")
        print("  - complete_summary_report.png")
        print("=" * 70)
    
    print("\n" + "=" * 70)
    print("TEST SET RESULTS SUMMARY")
    print("=" * 70)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Results saved to:")
    print(f"  - {test_output_file}")
    print(f"  - {test_csv}")
    
    print("\n" + "=" * 70)
    print("FISCHER LOVEBIRD HMM TRAINING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {PIPELINE_CFG.output_directory}")

    predicted_gender_labels = [result["predicted_label"] for result in detailed_results]
    actual_gender_labels = [result["true_label"] for result in detailed_results]

    return final_models, predicted_gender_labels, actual_gender_labels, detailed_results, test_results, test_accuracy


if __name__ == "__main__":
    train_gender_classification_hmm_models()
