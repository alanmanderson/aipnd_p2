import argparse

def get_train_input_args():
    parser = argparse.ArgumentParser(description="Neural net trainer")
    parser.add_argument('data_directory', type=str, help='directory to access the data')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='The directory where the data will be saved')
    parser.add_argument('--arch', type=str, choices=['alexnet', 'resnet', 'vgg13', 'vgg19'], default='vgg19', help='which architecture to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Sizes of hidden layer')
    parser.add_argument('--epochs', type=int, default=3, help='Number of Epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def get_predict_input_args():
    parser = argparse.ArgumentParser(description="Neural net trainer")
    parser.add_argument('filepath', type=str, help='image file path to be classified')
    parser.add_argument('checkpoint', type=str, help='')
    parser.add_argument('--top_k', type=int, help='The top K most likely classes')
    parser.add_argument('--category_names', type=str, help='json file with category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()
