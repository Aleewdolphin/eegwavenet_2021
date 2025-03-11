import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHDC(nn.Module):
    def __init__(self, in_features, out_features, encoder=None):
        super(LinearHDC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoder = encoder
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return torch.log_softmax(F.linear(x, self.weight), dim=-1)
    
    def fit(self, x, y, x_test, y_test, epochs=-1):
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.CrossEntropyLoss()
        print("'rWe are in LinearHDC right now")
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
    
    def predict(self, x):
        # pudb.set_trace()
        # xmean = ...
        return (self(x) - self(x).mean(axis=0)).numpy().argmax(1)