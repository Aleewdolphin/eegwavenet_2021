import os
from glob import glob
import csv
import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from epilepsy2bids.eeg import Eeg
from thuwajit_2021.utils import load_models, get_dataloader, predict
from tqdm import tqdm
from scipy import signal
from thuwajit_2021.LearningHD import LearningHD  # Import the training function
import json


def read_tsv(file_path):
    with open(file_path, 'r') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        data = []
        for row in tsvreader:
            data.append(row)
    return data

def create_y(file_path, fs, pre_seizure_sec=60, post_seizure_sec=60):
    """Create labels with exclusion zones around seizures."""
    try:
        data = read_tsv(file_path)
        del data[0]  # headers
        recording_duration = float(data[0][6])
        y = np.zeros(int(recording_duration * fs), dtype=np.int8)
        
        # First mark all seizures and exclusion zones
        seizure_periods = []
        for row in data:
            if row[2] == 'bckg':
                break
            try:
                # Ensure indices are integers
                onset = int(round(float(row[0]) * fs))
                duration = int(round(float(row[1]) * fs))
                
                if onset < len(y):
                    # Mark seizure
                    seizure_end = min(onset + duration, len(y))
                    y[onset:seizure_end] = 1
                    
                    # Store seizure period info
                    seizure_periods.append({
                        'onset': onset,
                        'end': seizure_end,
                        'duration': seizure_end - onset
                    })
            except (ValueError, IndexError) as e:
                print(f"Warning in file {file_path}, row {row}: {str(e)}")
                continue
        
        # Mark exclusion zones with -1
        exclusion_mask = np.zeros_like(y, dtype=bool)
        for period in seizure_periods:
            # Pre-seizure exclusion
            pre_start = max(0, period['onset'] - int(pre_seizure_sec * fs))
            exclusion_mask[pre_start:period['onset']] = True
            
            # Post-seizure exclusion
            post_end = min(len(y), period['end'] + int(post_seizure_sec * fs))
            exclusion_mask[period['end']:post_end] = True
        
        y[exclusion_mask] = -1
        
        return y, seizure_periods
    
    except Exception as e:
        print(f"Error in create_y for file {file_path}")
        print(f"Error details: {str(e)}")
        raise

def preprocess_eeg(data, fs, cutoff=64):
    """Apply low-pass filter to EEG data."""
    print("Applying low-pass filter...")
    b, a = signal.butter(4, cutoff, fs=fs, btype='low', analog=False)
    data_filtered = np.zeros_like(data)
    for channel in range(data.shape[0]):
        data_filtered[channel, :] = signal.filtfilt(b, a, data[channel, :])
    print("Filtering complete")
    return data_filtered

def get_file_pairs(base_dir):
    """Get matching EDF and TSV files."""
    file_pairs = []
    for subject_dir in glob(os.path.join(base_dir, 'sub-*')):
        for session_dir in glob(os.path.join(subject_dir, 'ses-*')):
            eeg_dir = os.path.join(session_dir, 'eeg')
            for edf_file in glob(os.path.join(eeg_dir, '*_eeg.edf')):
                tsv_file = edf_file.replace('_eeg.edf', '_events.tsv')
                if os.path.exists(tsv_file):
                    file_pairs.append((edf_file, tsv_file))
    return file_pairs

def process_file_generator(file_pairs, window_size_sec=4):
    """Generator that yields balanced windows and labels."""
    total_processed = 0
    n_channels = None
    
    for edf_file, tsv_file in tqdm(file_pairs, desc="Processing files"):
        try:
            # print(f"Processing EDF: {edf_file}")
            
            # Load EEG data
            eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
            if eeg.montage is Eeg.Montage.UNIPOLAR:
                eeg.reReferenceToBipolar()

            fs = eeg.fs
            window_size_samples = int(round(window_size_sec * fs))
            
            if n_channels is None:
                n_channels = eeg.data.shape[0]
                print(f"Using {n_channels} channels based on first file")
            elif eeg.data.shape[0] != n_channels:
                print(f"Warning: File {edf_file} has {eeg.data.shape[0]} channels, expected {n_channels}")
                continue
            
            # Get labels and seizure periods
            y, seizure_periods = create_y(tsv_file, fs)
            if not seizure_periods:
                # file {edf_file} - no seizures")
                continue
            
            # Preprocess data
            filtered_data = preprocess_eeg(eeg.data, fs)
            
            # For each seizure period, get equal amounts of seizure and non-seizure data
            for period in seizure_periods:
                seizure_windows = []
                non_seizure_windows = []
                
                # Get all seizure windows
                for i in range(period['onset'], period['end'] - window_size_samples + 1, window_size_samples):
                    if i + window_size_samples <= period['end']:
                        window = filtered_data[:, i:i + window_size_samples]
                        seizure_windows.append((window, 1))
                
                # Find valid non-seizure regions (y == 0)
                valid_regions = np.where(y == 0)[0]
                valid_starts = []
                
                # Group valid regions into windows
                i = 0
                while i < len(valid_regions):
                    if i + window_size_samples <= len(valid_regions):
                        # Check if this is a continuous window
                        window_indices = valid_regions[i:i + window_size_samples]
                        if np.all(np.diff(window_indices) == 1):  # Check continuity
                            valid_starts.append(window_indices[0])
                    i += window_size_samples
                
                # Randomly select equal number of non-seizure windows
                if valid_starts:
                    n_seizure = len(seizure_windows)
                    n_select = min(n_seizure, len(valid_starts))
                    selected_starts = np.random.choice(valid_starts, size=n_select, replace=False)
                    
                    for start in selected_starts:
                        window = filtered_data[:, start:start + window_size_samples]
                        non_seizure_windows.append((window, 0))
                
                # Yield balanced windows
                for sz_window, non_sz_window in zip(seizure_windows, non_seizure_windows):
                    yield torch.tensor(sz_window[0], dtype=torch.float32), sz_window[1]
                    yield torch.tensor(non_sz_window[0], dtype=torch.float32), non_sz_window[1]
            
            del eeg, filtered_data, y
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            total_processed += 1
            print(f"Successfully processed file {total_processed}")
                
        except Exception as e:
            print(f"\nError processing {edf_file}")
            print(f"Error details: {str(e)}")
            print(f"Skipping this file and continuing...")
            continue

def load_and_split_data(base_dir, test_size=0.2, random_state=42, save_dir=None):
    """Load and split data by taking test_size portion of each subject's files."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get all subjects and their files with run information
    subject_files = {}
    file_info = {}  # Store detailed info about each file
    
    for subject_dir in glob(os.path.join(base_dir, 'sub-*')):
        subject = os.path.basename(subject_dir)
        subject_files[subject] = []
        
        for session_dir in glob(os.path.join(subject_dir, 'ses-*')):
            session = os.path.basename(session_dir)
            eeg_dir = os.path.join(session_dir, 'eeg')
            
            for edf_file in glob(os.path.join(eeg_dir, '*_eeg.edf')):
                tsv_file = edf_file.replace('_eeg.edf', '_events.tsv')
                if os.path.exists(tsv_file):
                    # Extract run number from filename
                    filename = os.path.basename(edf_file)
                    run_info = filename.split('_run-')[1].split('_')[0] if '_run-' in filename else 'unknown'
                    
                    file_pair = (edf_file, tsv_file)
                    subject_files[subject].append(file_pair)
                    
                    # Store detailed file information
                    file_info[edf_file] = {
                        'subject': subject,
                        'session': session,
                        'run': run_info,
                        'full_name': filename,
                        'events_file': os.path.basename(tsv_file)
                    }
    
    # Split files for each subject
    train_pairs = []
    test_pairs = []
    split_info = {}  # Detailed split information
    
    print("\nSplitting files by subject:")
    for subject, files in subject_files.items():
        split_info[subject] = {
            'train_files': [],
            'test_files': [],
            'n_total': len(files)
        }
        
        n_files = len(files)
        n_test = max(1, int(n_files * test_size))
        
        # Randomly select test files
        np.random.seed(random_state)
        test_indices = np.random.choice(n_files, size=n_test, replace=False)
        test_mask = np.zeros(n_files, dtype=bool)
        test_mask[test_indices] = True
        
        # Split files and store detailed information
        for i, file_pair in enumerate(files):
            edf_file = file_pair[0]
            info = file_info[edf_file]
            
            if test_mask[i]:
                test_pairs.append(file_pair)
                split_info[subject]['test_files'].append(info)
            else:
                train_pairs.append(file_pair)
                split_info[subject]['train_files'].append(info)
        
        # Print detailed split information
        print(f"\nSubject {subject}:")
        print(f"├── Total files: {n_files}")
        print(f"├── Training files: {len(split_info[subject]['train_files'])}")
        print("│   └── Runs:", [f["run"] for f in split_info[subject]['train_files']])
        print(f"└── Test files: {len(split_info[subject]['test_files'])}")
        print("    └── Runs:", [f["run"] for f in split_info[subject]['test_files']])
    
    print(f"\nTotal split:")
    print(f"├── Total training files: {len(train_pairs)}")
    print(f"└── Total test files: {len(test_pairs)}")
    
    # Process data with balanced windows
    if save_dir:
        print("Processing and saving training data...")
        train_data_file = os.path.join(save_dir, 'x_train.npy')
        train_labels_file = os.path.join(save_dir, 'y_train.npy')
        
        # First pass to count windows
        n_train_windows = sum(1 for _ in process_file_generator(train_pairs))
        
        # Create memory-mapped files
        x_train = np.lib.format.open_memmap(train_data_file, mode='w+', 
                                          dtype=np.float32, 
                                          shape=(n_train_windows, 18, 1024))
        y_train = np.lib.format.open_memmap(train_labels_file, mode='w+',
                                          dtype=np.int8,
                                          shape=(n_train_windows,))
        
        # Fill files with preprocessed data
        for i, (window, label) in enumerate(process_file_generator(train_pairs)):
            x_train[i] = window.numpy()
            y_train[i] = label
        
        # Same for test data
        print("\nProcessing and saving test data...")
        test_data_file = os.path.join(save_dir, 'x_test.npy')
        test_labels_file = os.path.join(save_dir, 'y_test.npy')
        
        n_test_windows = sum(1 for _ in process_file_generator(test_pairs))
        
        x_test = np.lib.format.open_memmap(test_data_file, mode='w+',
                                         dtype=np.float32,
                                         shape=(n_test_windows, 18, 1024))
        y_test = np.lib.format.open_memmap(test_labels_file, mode='w+',
                                         dtype=np.int8,
                                         shape=(n_test_windows,))
        
        for i, (window, label) in enumerate(process_file_generator(test_pairs)):
            x_test[i] = window.numpy()
            y_test[i] = label
        
        # Save preprocessing parameters and split information
        with open(os.path.join(save_dir, 'preprocessing_info.txt'), 'w') as f:
            f.write("Preprocessing parameters:\n")
            f.write(f"Low-pass filter cutoff: 64 Hz\n")
            f.write(f"Filter order: 4\n")
            f.write(f"Window size: 4 seconds\n")
            f.write("\nTraining subjects:\n")
            f.write(",".join(list(split_info.keys())) + "\n")
            f.write("Test subjects:\n")
            f.write(",".join(list(split_info.keys())) + "\n")
        
        print(f"\nSaved processed data to {save_dir}")
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Save detailed split information
        split_info_file = os.path.join(save_dir, 'split_info.json')
        detailed_split = {
            'preprocessing_params': {
                'low_pass_cutoff': 64,
                'filter_order': 4,
                'window_size_sec': 4,
                'test_size': test_size,
                'random_state': random_state
            },
            'subjects': {}
        }
        
        for subject, info in split_info.items():
            detailed_split['subjects'][subject] = {
                'train_files': [
                    {
                        'run': f['run'],
                        'session': f['session'],
                        'filename': f['full_name'],
                        'events_file': f['events_file']
                    }
                    for f in info['train_files']
                ],
                'test_files': [
                    {
                        'run': f['run'],
                        'session': f['session'],
                        'filename': f['full_name'],
                        'events_file': f['events_file']
                    }
                    for f in info['test_files']
                ],
                'total_files': info['n_total']
            }
        
        with open(split_info_file, 'w') as f:
            json.dump(detailed_split, f, indent=2)
        
        print(f"\nSaved detailed split information to {split_info_file}")
        
        return x_train, y_train, x_test, y_test, split_info

def load_processed_data(save_dir):
    """Load the previously processed and saved data."""
    print("Loading data...")
    # Load with memory mapping
    x_train = np.load(os.path.join(save_dir, 'x_train.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(save_dir, 'y_train.npy'), mmap_mode='r')
    x_test = np.load(os.path.join(save_dir, 'x_test.npy'), mmap_mode='r')
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'), mmap_mode='r')
    
    # Convert to torch tensors with proper copying
    print("Converting to tensors...")
    x_train = torch.from_numpy(x_train.copy()).float()
    y_train = torch.from_numpy(y_train.copy()).long()
    x_test = torch.from_numpy(x_test.copy()).float()
    y_test = torch.from_numpy(y_test.copy()).long()
    
    # Load split information
    with open(os.path.join(save_dir, 'preprocessing_info.txt'), 'r') as f:
        lines = f.readlines()
        train_subjects = lines[-3].strip().split(",")
        test_subjects = lines[-1].strip().split(",")
    
    print("Data loading complete")
    return x_train, y_train, x_test, y_test, train_subjects, test_subjects

if __name__ == "__main__":
    base_dir = '/Users/annielee/Documents/research/seizure-only'
    save_dir = '/Users/annielee/Documents/research/seizure-only/balanced-data'
    
    # Process and save data
    x_train, y_train, x_test, y_test, split_info = load_and_split_data(
        base_dir, save_dir=save_dir
    )
    
    # # Or load previously processed data
    # x_train, y_train, x_test, y_test, train_subjects, test_subjects = load_processed_data(save_dir)
    
    # # Move to device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"\nUsing device: {device}")
    
    # # Print data shapes before training
    # print("Data shapes:")
    # print(f"x_train: {x_train.shape}")
    # print(f"y_train: {y_train.shape}")
    # print(f"x_test: {x_test.shape}")
    # print(f"y_test: {y_test.shape}")
    
    # # Print class distribution
    # print("\nClass distribution:")
    # print(f"Training - Seizure: {(y_train == 1).sum().item()} ({(y_train == 1).float().mean().item()*100:.1f}%)")
    # print(f"Testing - Seizure: {(y_test == 1).sum().item()} ({(y_test == 1).float().mean().item()*100:.1f}%)")
    
    # print("\nStarting model training...")
    # model = train_model(x_train, y_train, x_test, y_test, epochs=100, fold_number=1)
    
    # # 5. Evaluate
    # model.eval()
    # with torch.no_grad():
    #     test_predictions = model.predict(x_test)
    #     accuracy = (test_predictions == y_test).float().mean()
    #     print(f"Final Test Accuracy: {accuracy.item():.4f}")    