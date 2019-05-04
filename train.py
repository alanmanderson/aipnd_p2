from input.get_input_args import get_train_input_args
from get_dataloader import get_loader
from get_categories import get_cat_to_name
from torchvision import transforms
from torch import cuda
from pretrained_models import make_model
from checkpts import save_checkpoint
from checkpts import load_checkpoint
import trainer

args = get_train_input_args()
data_dir = args.data_directory
epochs = args.epochs
arch = args.arch
device = 'cpu'
if args.gpu and cuda.is_available():
    device = 'cuda'
hidden_layer_size = args.hidden_units
learning_rate = args.learning_rate
checkpoint_file = args.save_dir + "/checkpoint.pth"


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = {
    "training" : transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
    "validation" : transforms.Compose([transforms.Resize(224),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])
}
dataloaders = {
    "training" : get_loader(train_dir, data_transforms['training']),
    "validation" : get_loader(valid_dir, data_transforms['validation'])
}

cat_to_name = get_cat_to_name()
model = make_model(arch, hidden_layer_size, len(cat_to_name))

model = trainer.train(model, learning_rate, device, epochs, dataloaders['training'], dataloaders['validation'])

save_checkpoint(model, cat_to_name, arch, hidden_layer_size, epochs, checkpoint_file)
#model = load_checkpoint(checkpoint_file)

print(model)
