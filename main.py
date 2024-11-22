import torch
from torch import nn, optim
from data_preprocessing import get_data_loaders
from model import BrainTumorCNN
from train import train_model

def main():
    # Paths and configurations
    dataset_dir = "./data"
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(dataset_dir, batch_size=batch_size)
    num_classes = len(class_names)

    # Initialize model
    print("Initializing model...")
    model = BrainTumorCNN(num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Beginning Training...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

    # Save the trained model
    torch.save(trained_model.state_dict(), "brain_tumor_cnn.pth")
    print("Model saved to brain_tumor_cnn.pth")

if __name__ == "__main__":
    main()
