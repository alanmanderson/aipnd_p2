import torch
from pretrained_models import make_model

def save_checkpoint(model, class_to_idx, arch, hidden_layer, epochs, filepath):
    model.class_to_idx = class_to_idx
    model.to("cpu")
    checkpoint = {
        'arch': arch,
        'hidden_layer': hidden_layer,
        'state_dict': model.state_dict(),
        'class_to_idx' : class_to_idx,
        'out_size' : len(class_to_idx),
        'epochs': epochs
    }

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = make_model(checkpoint['arch'], checkpoint['hidden_layer'],
checkpoint['out_size'])
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
