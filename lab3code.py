import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


# Load MNIST Dataset in Numpy
def load_mnist_data():
    # 1000 training samples where each sample feature is a greyscale image with shape (28, 28)
    # 1000 training targets where each target is an integer indicating the true digit
    mnist_train_features = np.load('mnist_train_features.npy') 
    mnist_train_targets = np.load('mnist_train_targets.npy')

    # 100 testing samples + targets
    mnist_test_features = np.load('mnist_test_features.npy')
    mnist_test_targets = np.load('mnist_test_targets.npy')

    # Print the dimensions of training sample features/targets
    #print(mnist_train_features.shape, mnist_train_targets.shape)
    # Print the dimensions of testing sample features/targets
    #print(mnist_test_features.shape, mnist_test_targets.shape)
    
    return mnist_train_features, mnist_train_targets, mnist_test_features, mnist_test_targets


def flatten_features(features):
    # Flatten the features from (28, 28) to (784,)
    return features.reshape(features.shape[0], -1)

def scale_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# More general function to load datasets, including solar_flare
def load_dataset(dataset_name: str = ""):
    if dataset_name == "mnist":
        train_features, train_targets, test_features, test_targets = load_mnist_data()
        train_features = flatten_features(train_features)
        test_features = flatten_features(test_features)
        train_features = scale_features(train_features)
        test_features = scale_features(test_features)
        
    elif dataset_name == "solar_flare":
        # Load the solar flare dataset
        solar_flare = fetch_ucirepo(id=89)
        
        # Simplifying slightly for the sake of this example
        solar_flare.data.targets = solar_flare.data.targets['severe flares']
        
        # Split the solar flare dataset into train and test sets (90:10 split)
        train_features, test_features, train_targets, test_targets = train_test_split(
            solar_flare.data.features, solar_flare.data.targets, test_size=0.1, random_state=42
        )
        
        # Onehot encode modified Zurich class, largest spot size, spot distribution
        onehot_columns = ["modified Zurich class", "largest spot size", "spot distribution"]
        for col in onehot_columns:
            onehot = pd.get_dummies(train_features[col], prefix=col)
            train_features = pd.concat([train_features, onehot], axis=1)
            train_features.drop(col, axis=1, inplace=True)
            
            onehot = pd.get_dummies(test_features[col], prefix=col)
            test_features = pd.concat([test_features, onehot], axis=1)
            test_features.drop(col, axis=1, inplace=True)
        
        # Scale the features
        train_features = scale_features(train_features)
        test_features = scale_features(test_features)
        
        # Convert targets to numpy arrays
        train_targets = train_targets.to_numpy()
        test_targets = test_targets.to_numpy()
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # train-test split
    train_features, val_features, train_targets, val_targets = train_test_split(train_features, train_targets, test_size=0.2)
    
    return train_features, train_targets, val_features, val_targets, test_features, test_targets


# Train
def train_model(model, train_features, train_targets, validation_features, validation_targets, 
                test_features=None, test_targets=None, learning_rate=0.0015, epochs=80, batch_size=64):
    """
    Train a neural network model on the provided data.
    
    Parameters:
        model: PyTorch model to train
        train_features: Training features as numpy array
        train_targets: Training targets as numpy array
        validation_features: Validation features as numpy array
        validation_targets: Validation targets as numpy array
        test_features: Test features as numpy array (optional)
        test_targets: Test targets as numpy array (optional)
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        tuple: (trained model, training loss list, validation accuracy list)
    """
    # Initialize tracking lists
    train_loss_list = np.zeros(epochs)
    validation_accuracy_list = np.zeros(epochs)
    
    # Convert numpy arrays to PyTorch tensors
    train_inputs = torch.from_numpy(train_features).float()
    train_targets = torch.from_numpy(train_targets).long()
    
    validation_inputs = torch.from_numpy(validation_features).float()
    validation_targets = torch.from_numpy(validation_targets).long()
    
    if test_features is not None and test_targets is not None:
        test_inputs = torch.from_numpy(test_features).float()
        test_targets = torch.from_numpy(test_targets).long()
        test_dataset = TensorDataset(test_inputs, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = TensorDataset(validation_inputs, validation_targets)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer and scheduler
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        train_inputs = train_inputs.cuda()
        validation_inputs = validation_inputs.cuda()
        validation_targets = validation_targets.cuda()
    
    # Training Loop
    for epoch in tqdm.trange(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for batch_inputs, batch_targets in train_loader:
            if torch.cuda.is_available():
                batch_inputs, batch_targets = batch_inputs.cuda(), batch_targets.cuda()
            
            optimizer.zero_grad()  # Reset gradients to zero
            outputs = model(batch_inputs)  # Forward pass with current batch
            loss = loss_func(outputs, batch_targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item() * batch_inputs.size(0)
        
        # Store average epoch loss
        train_loss_list[epoch] = running_loss / len(train_dataset)
        scheduler.step()  # Update learning rate with cosine annealing
        
        # Compute Validation Accuracy
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            correct = 0
            total = 0
            for val_inputs, val_targets in validation_loader:
                if torch.cuda.is_available():
                    val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()
                outputs = model(val_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += val_targets.size(0)
                correct += (predicted == val_targets).sum().item()
            
            validation_accuracy_list[epoch] = correct / total
    
    # Compute test accuracy if test data is provided
    test_accuracy = None
    if test_features is not None and test_targets is not None:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for test_inputs, test_targets in test_loader:
                if torch.cuda.is_available():
                    test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
                outputs = model(test_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += test_targets.size(0)
                correct += (predicted == test_targets).sum().item()
            
            test_accuracy = correct / total
    
    return model, train_loss_list, validation_accuracy_list, test_accuracy


# Visualize and evaluate
def visualize_training(train_loss_list, validation_accuracy_list):
    """
    Visualize training loss and validation accuracy.
    
    Parameters:
        train_loss_list: List of training losses
        validation_accuracy_list: List of validation accuracies
    """
    plt.figure(figsize = (12, 6))

    # Visualize training loss with respect to iterations (1 iteration -> single batch)
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_list, linewidth = 3)
    plt.ylabel("training loss")
    plt.xlabel("epochs")
    sns.despine()

    # Visualize validation accuracy with respect to epochs
    plt.subplot(2, 1, 2)
    plt.plot(validation_accuracy_list, linewidth = 3, color = 'gold')
    plt.ylabel("validation accuracy")
    sns.despine()
    
    plt.show()

