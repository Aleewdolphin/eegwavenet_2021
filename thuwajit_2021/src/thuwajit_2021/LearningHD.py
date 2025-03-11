import torch
from learncoder import Encoder
from NN import LinearHDC
import torch.nn as nn
from tqdm import tqdm, trange
import time
import numpy as np
from collections import Counter

import pudb

class LearningHD(nn.Module):
    def __init__(self, classes, features, dim=10000):
        super(LearningHD, self).__init__()
        # print("\nInitializing LearningHD components...")
        # print("├── Creating Encoder...")
        self.encoder = Encoder(features, dim)
        # print("├── Creating LinearHDC classifier...")
        self.model = LinearHDC(dim, classes)
        print(f"└── Done! Encoder basis dtype: {self.encoder.basis.dtype}")
    
    def forward(self, x):
        # First pass through encoder
        # print("Passing to encoder")
        encoded = self.encoder(x)
        
        # Then through classifier
        output = self.model(encoded)
        return output
    
    def encode(self, x):
        print(f"Encoding data batch of shape {x.shape}...")
        return self.encoder(x)
    
    def fit(self, x_path, y_path, x_test_path, y_test_path, epochs=30):
        print("\nStarting training (fit function)...")

        x_data = np.load(x_path, mmap_mode='r')
        y_data = np.load(y_path, mmap_mode='r')
        x_test_data = np.load(x_test_path, mmap_mode='r')
        y_test_data = np.load(y_test_path, mmap_mode='r')
        batch_size = 64

        x_shape = np.load(x_path, mmap_mode='r').shape
        y_shape = np.load(y_path, mmap_mode='r').shape
        x_test_shape = np.load(x_test_path, mmap_mode='r').shape
        y_test_shape = np.load(y_test_path, mmap_mode='r').shape

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        x = torch.from_numpy(x_data.copy()).float()
        y = torch.from_numpy(y_data.copy()).long()
        x_test = torch.from_numpy(x_test_data.copy()).float()
        y_test = torch.from_numpy(y_test_data.copy()).long()
        
        epoch = 0
        last_loss = float('inf')
        while True:
            self.train()
            optimizer.zero_grad()
            output = self(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            self.eval()
            with torch.no_grad():
                output = self(x_test)
                test_loss = criterion(output, y_test)
                # get test accuracy
                y_pred = self.predict(x_test)
                acc = (y_pred == y_test).float().mean().item()
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item()} - Test Loss: {test_loss.item()} - Test Accuracy: {acc}')
            
            epoch += 1
            if epochs > 0 and epoch >= epochs:
                break

            if -1e-6 < (last_loss - loss.item()) < 1e-6:
                break
            last_loss = loss.item()
        return self

        # optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        # criterion = nn.CrossEntropyLoss()
        
        # # Progress bar for epochs
        # epoch_bar = trange(epochs, desc='Training Progress')
        # last_loss = float('inf') 
        
        # for epoch in epoch_bar:
        #     # Training phase
        #     self.train()
        #     n_batches = 0
        #     total_loss = 0
        #     test_loss = 0

        #     for i in range(0, x_shape[0], batch_size):
        #         x = torch.from_numpy(x_data[i:i+batch_size].copy()).float()
        #         y = torch.from_numpy(y_data[i:i+batch_size].copy()).long()
        #         optimizer.zero_grad()

        #         output = self(x)
            
        #         loss = criterion(output, y)
        #         loss.backward()
            
        #         optimizer.step()
        #         total_loss += loss.item()
        #         n_batches += 1

        #         print(f'\rBatch {n_batches} has loss of {loss.item()}')

        #         del x, y, output, loss
            
        #     # Evaluation phase
        #     self.eval()
        #     with torch.no_grad():
        #         for i in range(0, x_test_shape[0], batch_size):
        #             x_test = torch.from_numpy(x_test_data[i:i+batch_size].copy()).float()
        #             y_test = torch.from_numpy(y_test_data[i:i+batch_size].copy()).long()
        #             test_output = self(x_test)
        #             test_loss = criterion(test_output, y_test)
        #             y_pred = self.predict(x_test)
        #             acc = (y_pred == y_test).float().mean().item()
                
        #             # Update progress bar
        #             epoch_bar.set_postfix({
        #             'train_loss': f'{total_loss/n_batches:.4f}',
        #             'test_loss': f'{test_loss.item():.4f}',
        #             'test_acc': f'{acc:.4f}',
        #             })

        #             del x_test, y_test, test_output, y_pred
            
        #     # Early stopping check
        #     if epoch > 0 and abs(total_loss/n_batches - last_loss) < 1e-6:
        #         print("\nConverged! Stopping early...")
        #         break
        #     last_loss = total_loss/n_batches
        # print("\nTraining completed!")
        # return self
    
    def predict(self, x):
        # pudb.set_trace()
        pred= self(x).argmax(1)

        # pred =  (self(x) - self(x).mean(axis=0)).numpy().argmax(1)
        from collections import Counter
        print(Counter(list(pred)))
        return pred