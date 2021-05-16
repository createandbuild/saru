import sys
from tqdm import tqdm
import numpy as np
import cv2
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
import PIL
import pickle

from facenet_pytorch import MTCNN, InceptionResnetV1

# https://pytorch.org/docs/stable/_modules/torch/nn/modules/distance.html#CosineSimilarity
class CosineDistance(nn.Module):
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineDistance, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 1 - F.cosine_similarity(x1, x2, self.dim, self.eps)

class RecognitionModel():
    # structure borrowed from https://github.com/ppwwyyxx/Adversarial-Face-Attack

    def __init__(self):
        self.face_detect = MTCNN(
                image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                keep_all=True,
                device='cpu')
        self.network = InceptionResnetV1(pretrained='vggface2').eval()

    def compute_target_embedding(self, path):
        assert os.path.isdir(path), path
        images = glob.glob(os.path.join(path, '*.jpg'))
        image_batch = [cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1] for f in images]
        embeddings = self.eval_embeddings(image_batch)
        self.target_embeddings = embeddings
        return embeddings

    def pgd_attack(self, input_image, eps=16/255, alpha=3/255, steps=40):

        # https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py

        adv_image = input_image.clone().detach()
        criterion = CosineDistance()

        print("Beginning PGD attack ===>")
        for i in tqdm(range(steps)):
            adv_image.requires_grad = True
            image_embedding = self.network(adv_image)
            criterion = CosineDistance()

            loss = criterion(image_embedding, self.target_embeddings)
            grad = torch.autograd.grad(loss, adv_image, retain_graph=False, \
                    create_graph=False)[0]

            adv_image = adv_image.detach() - alpha*grad.sign()
            delta = torch.clamp(adv_image - input_image, min=-eps, max=eps)
            adv_image = torch.clamp(input_image + delta, min=-1, max=1).detach()

        return adv_image


    def perturb_all_faces(self, input_image):
        new_image = np.array(input_image)
        faces_cropped = self.face_detect(input_image)
        boxes, probs = self.face_detect.detect(input_image)

        for i, box in enumerate(boxes):
            perturbed_face = self.pgd_attack(faces_cropped[i].unsqueeze(0))
            width = int(box[2]-box[0])
            height = int(box[3]-box[1])

            resize = transforms.Resize((height, width))
            resized_face = resize(perturbed_face)
            resized_face = (resized_face.squeeze(0) * 128 + 127.5).permute(1, 2, 0).numpy()

            new_image[int(box[1]): int(box[1])+height, int(box[0]): int(box[0])+width] = resized_face

        return new_image

    def eval_embeddings(self, batch_array):
        embedding_list = []

        for img in batch_array:
            face_cropped = self.face_detect(img)
            self.network.eval()
            img_embedding = self.network(face_cropped).detach()
            embedding_list.append(img_embedding)

        return torch.mean(torch.stack(embedding_list), dim=0)

    def distance_to_target(self, img):
        emb = self.eval_embeddings([img])
        dist = np.dot(emb, self.target_embeddings.T).flatten()
        stats = np.percentile(dist, [10, 30, 50, 70, 90])
        return stats

    def detect_people(self, image, out_path):
        # based off of https://github.com/timesler/facenet-pytorch
        faces_cropped = self.face_detect(image)
        identities = []

        boxes, _ = self.face_detect.detect(image)

        image_draw = image.copy()
        draw = PIL.ImageDraw.Draw(image_draw)

        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        for i in range(len(faces_cropped)):
            input_face = faces_cropped[i].unsqueeze(0)
            img_embedding = self.network(input_face).detach()

            threshold = 0.40
            criterion = torch.nn.CosineSimilarity()

            lowest_dist = 1000
            curr_name = 'No Match'

            for name, gt_embedding in zip(self.person_embeddings.keys(), self.person_embeddings.values()):
                loss = criterion(img_embedding, gt_embedding)
                distance = 1 - loss
                if distance <= threshold:
                    if lowest_dist > distance:
                        curr_name = name
                        lowest_dist = min(lowest_dist, distance)

            identities.append({"name": curr_name, "bbox": boxes[i]})

        font = PIL.ImageFont.truetype("Roboto-Regular.ttf", 40)
        for identity in identities:
            draw.text((identity["bbox"][0], identity["bbox"][1]-50), identity["name"],(255,0,0),font=font)

        image_draw = image_draw.save(out_path)

        return identities


    def test_adv_image(self, image, gt_image, orig_face_idx,  name):
        faces_cropped = self.face_detect(image)
        gt_cropped = self.face_detect(gt_image)

        image_np = (faces_cropped[orig_face_idx] * 128 + 127.5).permute(1, 2, 0).numpy()
        cv2.imwrite('data/kim/input.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        faces_cropped[orig_face_idx] = self.pgd_attack(faces_cropped[orig_face_idx].unsqueeze(0))

        adv_np = (faces_cropped[orig_face_idx] * 128 + 127.5).permute(1, 2, 0).numpy()
        cv2.imwrite('data/kim/output.jpg', cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR))

        for i in range(len(faces_cropped)):
            input_face = faces_cropped[i].unsqueeze(0)
            img_embedding = self.network(input_face).detach()
            gt_embedding = self.network(gt_cropped).detach()

            threshold = 0.40
            criterion = torch.nn.CosineSimilarity()
            loss = criterion(img_embedding, gt_embedding)
            distance = 1 - loss
            print(distance)
            if distance <= threshold:
                print('Detected {} in the image!'.format(name))
                return True
            else:
                print('No match!')

    def build_database(self, gt_path, out_path):
        assert os.path.isdir(gt_path), gt_path
        images = glob.glob(os.path.join(gt_path, '*.jpg'))

        person_embeddings = {}

        for f in images:
            gt = cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1]
            name = ' '.join(os.path.basename(f).split('.')[0].split('_'))

            face_cropped = self.face_detect(gt)
            img_embedding = self.network(face_cropped).detach()
            person_embeddings[name] = img_embedding

        with open(os.path.join(out_path, 'person_embeddings.dat'), 'wb') as outfile:
            pickle.dump(person_embeddings, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        return person_embeddings

    def load_database(self, data_path):

        with open(os.path.join(data_path, "person_embeddings.dat"), 'rb') as infile:
            self.person_embeddings = pickle.load(infile)


def attack(path):
    model = RecognitionModel()
    #model.build_database("data/ground_truths", "data")
    model.load_database("data")

    target_embeddings = model.compute_target_embedding('data/mlk')
    img = PIL.Image.open(path)

    identities = model.detect_people(img, out_path="data/exports/out_clean.jpg")
    defended_image = model.perturb_all_faces(img)
    cv2.imwrite('data/exports/defended.jpg', cv2.cvtColor(defended_image, cv2.COLOR_RGB2BGR))

    testing = PIL.Image.open("data/exports/defended.jpg")
    adv_identities = model.detect_people(testing, out_path="data/exports/out_adv.jpg")


