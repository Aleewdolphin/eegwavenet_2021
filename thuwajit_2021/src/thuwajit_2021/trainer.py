from time import time
import torch
import numpy as np
import pickle
import os
from thuwajit_2021.LearningHD import LearningHD
from torch.utils.data import DataLoader, TensorDataset
# import pudb

def load_data_in_batches(file_path, batch_size=32):
    """Load data in batches using memory mapping"""
    data = np.load(file_path, mmap_mode='r')
    for i in range(0, len(data), batch_size):
        batch = torch.from_numpy(data[i:i+batch_size].copy()).float()
        yield batch

def main(dataset: str, dir='data', dim=4000, args=None):
    # Load data paths
    print('Loading...')
    if dataset == 'seizure':
        save_dir = '/Users/annielee/Documents/research/seizure-only/balanced-data'
        x_path = os.path.join(save_dir, 'x_train.npy')
        y_path = os.path.join(save_dir, 'y_train.npy')
        x_test_path = os.path.join(save_dir, 'x_test.npy')
        y_test_path = os.path.join(save_dir, 'y_test.npy')
        
        # Get data shapes without loading
        x_shape = np.load(x_path, mmap_mode='r').shape
        y_shape = np.load(y_path, mmap_mode='r').shape
        x_test_shape = np.load(x_test_path, mmap_mode='r').shape
        y_test_shape = np.load(y_test_path, mmap_mode='r').shape
    else:
        # Original data loading for other datasets
        load = getattr(__import__(dataset), 'load')
        x, x_test, y, y_test = load()

    # Device setup
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f'Using NVIDIA GPU: {torch.cuda.get_device_name()}.')
    else:
        device = 'cpu'
        print('Using CPU.')

    torch.manual_seed(0)
    np.random.seed(0)

    # Get dimensions and classes
    if dataset == 'seizure':
        # Load a small batch to get unique classes
        y_sample = np.load(y_path, mmap_mode='r')[:1000]
        classes = len(np.unique(y_sample))
        n_channels = x_shape[1]  # 18 channels
        n_timepoints = x_shape[2]  # 1024 timepoints
        print(f'Input shape: {x_shape}')
    else:
        features = x.size(1)
        classes = len(torch.unique(y))
        print(f'Features: {features}')

    # Model initialization
    if dataset == 'seizure':
        model = LearningHD(classes=classes, features=n_channels, dim=dim)
    else:
        model = LearningHD(classes, features, dim)
    
    model = model.to(device)

    # Print shapes
    print(f'Classes: {classes}')
    if dataset == 'seizure':
        print(f'x shape: {x_shape}')
        print(f'x_test shape: {x_test_shape}')
        # pudb.set_trace()
        print(f'y shape: {y_shape}')
        print(f'y_test shape: {y_test_shape}')
    else:
        print(f'x shape: {x.shape}')
        print(f'x_test shape: {x_test.shape}')
        print(f'y shape: {y.shape}')
        print(f'y_test shape: {y_test.shape}')

    print('Training...')
    if dataset == 'seizure':
        model.fit(x_path, y_path, x_test_path, y_test_path, epochs=args.epochs)
    
    else:
        # Original training code for other datasets
        model.fit(x, y, x_test, y_test, epochs=args.epochs)
    
    print('Training completed.')
    print('Testing... FINAL TESTING')

    # Final evaluation
    x_test_data = np.load(x_test_path, mmap_mode='r')
    y_test_data = np.load(y_test_path, mmap_mode='r')
    x_test = torch.from_numpy(x_test_data.copy()).float()
    y_test = torch.from_numpy(y_test_data.copy()).long()
    y_pred = model.predict(x_test)
    print(f'y_pred: {y_pred}')
    acc = (y_pred == y_test).float().mean().item()
    num_correct = (y_pred == y_test).sum().item()
    num_total = y_test.size(0)
    print(f'Accuracy: {acc} ({num_correct} / {num_total})')

    # Save directory
    if not os.path.exists(dir):
        os.makedirs(dir)

    print('Exporting model...', end='')
    
    # Save model weights
    if dataset == 'seizure':
        class_hvs = model.model.weight.cpu().detach().numpy()
        encoder_basis = model.encoder.basis.cpu().detach().numpy()
        encoder_base = model.encoder.base.cpu().detach().numpy()
        
        with open(f'{dir}/model.pkl', 'wb') as f:
            pickle.dump(class_hvs, f)
        with open(f'{dir}/encoder_basis.pkl', 'wb') as f:
            pickle.dump(encoder_basis, f)
        with open(f'{dir}/encoder_base.pkl', 'wb') as f:
            pickle.dump(encoder_base, f)
    print('done.')

    # Export encoded data
    print('Exporting encoded data...', end='')

    if args.export_mean or args.find_diff_threshold or args.find_absolute_threshold:
        print("\nSkipping threshold calculations for memory efficiency")
        print("Please run these separately if needed")

    print('done.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--trainer', type=str, required=True)
    parser.add_argument('--dir', type=str, default='seizure_models')
    parser.add_argument('--dim', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-b', "--binary", action='store_true')
    parser.add_argument('--fd', type=int, default=100)
    parser.add_argument('--vd', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--export_mean', help="Export mean thresholds for termination heuristic 'mean'", action='store_true', default=False)
    parser.add_argument('--find_diff_threshold', help="Find threshold for termination heuristic 'diff'", action='store_true', default=False)
    parser.add_argument('--find_absolute_threshold', help="Find threshold for termination heuristic 'absolute'", action='store_true', default=False)
    args = parser.parse_args()

    main(args.dataset, args.dir, args.dim, args)
