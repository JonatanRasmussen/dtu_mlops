# Filename: main.py
import torch
from torch import nn, optim
from model import MyAwesomeModel
from data import mnist

MODEL_CHECKPOINT_FILENAME = "model_checkpoint.pth"

def train(lr=1e-3, epochs=10, batch_size=64):
    print(f"Training model with learning rate: {lr}, epochs: {epochs}, batch size: {batch_size}")

    model = MyAwesomeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load the data
    (train_images, train_targets), _ = mnist()

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(train_images, train_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.shape[0], -1)  # Reshape the images

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    # Save the trained model checkpoint
    torch.save(model.state_dict(), MODEL_CHECKPOINT_FILENAME)
    evaluate(MODEL_CHECKPOINT_FILENAME)


def evaluate(model_checkpoint):
    print(f"Evaluating model: {model_checkpoint}")

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    # Load the test data
    _, (test_images, test_targets) = mnist()  # Fix here to unpack the test images and labels

    correct = 0
    total = 0
    with torch.no_grad():
        # Reshape test_images to (number_of_samples, 784)
        test_images = test_images.view(-1, 784)  # Fix here to reshape the images

        outputs = model(test_images)
        _, predicted = torch.max(outputs, 1)
        total += test_targets.size(0)
        correct += (predicted == test_targets).sum().item()

    print(f'Accuracy of the network on test images: {100 * correct // total} %')


def main():
    choice = input("Type 't' to train or 'e' to evaluate: ").strip().lower()

    if choice == 't':
        use_defaults = input("Use default parameters [y/n]? ").strip().lower()
        if use_defaults == 'y':
            train()
        else:
            lr = float(input("Enter learning rate (default 1e-3): ") or 1e-3)
            epochs = int(input("Enter number of epochs (default 10): ") or 10)
            batch_size = int(input("Enter batch size (default 64): ") or 64)
            train(lr=lr, epochs=epochs, batch_size=batch_size)
    elif choice == 'e':
        evaluate(MODEL_CHECKPOINT_FILENAME)
    else:
        print("Invalid input. Please type 't' to train or 'e' to evaluate.")

if __name__ == "__main__":
    main()

#How to Run the Scripts:
# Training: The following will train the model and save the checkpoint.
# python s1_development_environment\exercise_files\final_exercise\main.py train
# python main.py train --lr 1e-3

# Evaluation: This will evaluate the saved model on the test set.
# python s1_development_environment\exercise_files\final_exercise\main.py evaluate model_checkpoint.pth
# python main.py evaluate model_checkpoint.pth