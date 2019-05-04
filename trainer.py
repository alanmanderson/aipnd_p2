import torch
from torch import nn
from torch import optim

def validate(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        #images.resize_(images.shape[0], 784)
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def train(model, learning_rate, device, epochs, training_loader, validation_loader):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    schedule = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 40

    for e in range(epochs):
        model.train()
        for images, labels in training_loader:
            images, labels = images.to(device), labels.to(device)
            steps += 1

            # Flatten images into a 25088 long vector
            #images.resize_(images.size()[0], 25088)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % 10 == 0: print(steps)

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validate(model, validation_loader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(validation_loader)),
                       "Test Accuracy: {:.3f}".format(accuracy/len(validation_loader)))

                running_loss = 0

            # Make sure training is back on
            model.train()
    return model
