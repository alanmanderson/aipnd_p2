from input.get_input_args import get_predict_input_args
from checkpts import load_checkpoint
from predictor import predict

from torch import cuda

args = get_predict_input_args()

image_path = args.filepath
checkpoint = args.checkpoint

top_k = args.top_k
category_file = args.category_names
device = 'cpu'
if args.gpu and cuda.is_available():
    device = 'cuda'

model = load_checkpoint(checkpoint)

top_p, classes = predict(image_path, model, top_k, device)

for i in range(len(classes)):
    print("Class: " + classes[i] + "Probability: " + str(top_p[i]))
