# Import necessary libraries
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network module
import torch.optim as optim  # Optimization algorithms
import torchvision  # Computer vision datasets and models
from torchvision import datasets, transforms  # Datasets and image transformations
import numpy as np  # Numerical computing
import sys  # System-specific parameters and functions
from PIL import Image  # Python Imaging Library for image processing

# Define hyperparameters for the neural network
batch_size = 64  # Number of samples processed in each training iteration
learning_rate = 0.001  # Step size for gradient descent
num_epochs = 10  # Number of complete passes through the training dataset
input_size = 28 * 28  # Size of input images (28x28 pixels flattened)
hidden_size = 128  # Number of neurons in hidden layers
num_classes = 10  # Number of output classes (10 fashion categories)

# Define directory where the dataset is stored
DATA_DIR = "."  # Current directory
download_dataset = False  # Don't download dataset (it should already be present)

# Try to load the FashionMNIST dataset
try:
    # Load training data with ToTensor transformation (converts images to normalized tensors)
    train_dataset = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset, transform=transforms.ToTensor())
    # Load test data with the same transformation
    test_dataset = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset, transform=transforms.ToTensor())
except Exception as e:
    # Handle any errors that occur during dataset loading
    print(f"Error loading dataset: {e}")
    print("Make sure the FashionMNIST folder is in the current directory.")
    sys.exit(1)  # Exit with error code

# Define class labels that correspond to the FashionMNIST dataset indices
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Split training data into training and validation sets
train_size = int(0.8 * len(train_dataset))  # Use 80% for training
val_size = len(train_dataset) - train_size  # Use 20% for validation
# Create random subsets for training and validation
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create data loaders for batched training and evaluation
# Training loader with shuffling to randomize the order of samples
train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
# Validation loader without shuffling as order doesn't matter for evaluation
val_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
# Test loader also without shuffling
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model for FashionMNIST classification
class FashionMNISTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the neural network layers
        
        Args:
            input_size: Size of input features (28*28 = 784)
            hidden_size: Size of hidden layers
            num_classes: Number of output classes (10)
        """
        super(FashionMNISTClassifier, self).__init__()  # Initialize the parent class
        self.flatten = nn.Flatten()  # Layer to flatten the input images
        
        # Define the neural network layers as a sequential model
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # First linear layer: input_size -> hidden_size
            nn.ReLU(),  # ReLU activation function for non-linearity
            nn.Dropout(0.2),  # Dropout with 20% probability to prevent overfitting
            nn.Linear(hidden_size, hidden_size),  # Second linear layer: hidden_size -> hidden_size
            nn.ReLU(),  # ReLU activation for second layer
            nn.Dropout(0.2),  # Dropout for second layer
            nn.Linear(hidden_size, num_classes)  # Output layer: hidden_size -> num_classes
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        x = self.flatten(x)  # Flatten input from [batch_size, 1, 28, 28] to [batch_size, 784]
        return self.model(x)  # Pass through the sequential model

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model and move it to the selected device
model = FashionMNISTClassifier(input_size, hidden_size, num_classes).to(device)
# Define the loss function (Cross Entropy Loss is standard for classification)
criterion = nn.CrossEntropyLoss()
# Define the optimizer (Adam is an adaptive learning rate optimizer)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to train the model
def train_model():
    """
    Train the neural network model on the FashionMNIST dataset
    """
    print("Training started...")
    # Open a log file to record training progress
    log_file = open("log.txt", "w")
    
    # Loop through each epoch
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode (enables dropout)
        running_loss = 0.0  # Initialize loss counter for this epoch
        
        # Loop through each batch in the training data
        for i, (images, labels) in enumerate(train_loader):
            # Move the current batch to the device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass: compute predictions
            outputs = model(images)
            # Calculate loss between predictions and true labels
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            
            # Accumulate loss for reporting
            running_loss += loss.item()
        
        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        # Print and log epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}\n')
        
        # Calculate validation accuracy for this epoch
        val_accuracy = evaluate_model(val_loader)
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        log_file.write(f'Validation Accuracy: {val_accuracy:.2f}%\n')
    
    # Calculate final test accuracy
    test_accuracy = evaluate_model(test_loader)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    log_file.write(f'Test Accuracy: {test_accuracy:.2f}%\n')
    log_file.close()  # Close the log file
    
    # Save the trained model to disk
    torch.save(model.state_dict(), 'fashion_mnist_model.pth')
    print("Done!")

# Function to evaluate the model on a given dataset
def evaluate_model(data_loader):
    """
    Evaluate the model's accuracy on a given dataset
    
    Args:
        data_loader: DataLoader containing the dataset to evaluate
        
    Returns:
        float: Accuracy percentage
    """
    model.eval()  # Set model to evaluation mode (disables dropout)
    correct = 0  # Initialize counter for correct predictions
    total = 0  # Initialize counter for total samples
    
    with torch.no_grad():  # Disable gradient computation to save memory
        # Loop through each batch in the dataset
        for images, labels in data_loader:
            # Move the current batch to the device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass: compute predictions
            outputs = model(images)
            # Get the predicted class (highest probability)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counters
            total += labels.size(0)  # Add batch size to total
            correct += (predicted == labels).sum().item()  # Add number of correct predictions
    
    # Calculate and return accuracy percentage
    return 100 * correct / total

# Function to classify a single image from a JPG file
def classify_image(image_path):
    """
    Classify a single image from a file path
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Predicted class label
    """
    try:
        # Load the image using torchvision's read_image function
        img = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.GRAY)
        img = img.float() / 255.0  # Normalize pixel values to [0, 1]
        
        # Ensure the image is the right size (28x28)
        if img.shape[1:] != (28, 28):
            # Create a transformation pipeline if resizing is needed
            transform = transforms.Compose([
                transforms.ToPILImage(),  # Convert to PIL Image
                transforms.Resize((28, 28)),  # Resize to 28x28
                transforms.ToTensor()  # Convert back to tensor
            ])
            img = transform(img)  # Apply transformations
        
        # Make prediction
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            img = img.to(device)  # Move image to device
            output = model(img.unsqueeze(0))  # Add batch dimension and forward pass
            _, predicted = torch.max(output, 1)  # Get predicted class
            class_name = class_labels[predicted.item()]  # Get class label
            
        return class_name
    except Exception as e:
        # Handle any errors that occur during image processing
        return f"Error processing image: {e}"

# Main execution block
if __name__ == "__main__":
    # Train the model
    train_model()
    
    # Interactive classification loop
    while True:
        # Get file path from user
        file_path = input("Please enter a filepath: ")
        
        # Check if user wants to exit
        if file_path.lower() == 'exit':
            print("Exiting...")
            break
        
        # Classify the image and print result
        prediction = classify_image(file_path)
        print(f"Classifier: {prediction}")