from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    keep_all=True,
    device=device
)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image
import cv2

img = Image.open("data/kim/testing/9.jpg")
ground_truth = Image.open("data/kim/ground_truth.jpg")

#https://github.com/serengil/deepface/blob/3b922eb7edab28fdec46349795a98288708d899d/deepface/commons/distance.py
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(source_representation, np.transpose(test_representation))
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# Get cropped and prewhitened image tensor
faces_cropped = mtcnn(img)
gt_cropped = mtcnn(ground_truth)

# Calculate embedding (unsqueeze to add batch dimension)
for i in range(len(faces_cropped)):
    img_embedding = resnet(faces_cropped[i].unsqueeze(0)).detach().numpy()
    gt_embedding = resnet(gt_cropped).detach().numpy()

    threshold = 0.40
    criterion = torch.nn.CosineSimilarity()
    loss = criterion(torch.Tensor(img_embedding), torch.Tensor(gt_embedding))
    distance = 1 - loss
    print(distance)
    if distance <= threshold:
        print('This is Kim!')

