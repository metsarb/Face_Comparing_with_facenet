import ssl
import certifi
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2

ssl_context = ssl.create_default_context(cafile = certifi.where())
ssl.create_default_https_context = lambda: ssl_context

device = torch.device('cpu')

mtcnn = MTCNN(image_size = 160, margin = 0, min_face_size = 30)
resnet = InceptionResnetV1(pretrained = 'vggface2').eval()

def load_and_preprocess_image(image_path, mtcnn):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Image loaded: {img.shape}")
    print(f"Image converted to RGB: {img_rgb.shape}")

    face = mtcnn(img_rgb)
    if face is not None:
        return face.unsqueeze(0)
    return None

def get_embedding(face,resnet):
    with torch.no_grad():
        embedding = resnet(face)
    return embedding


def calculate_l2_distance(embedding1, embedding2):
    return torch.norm(embedding1 - embedding2).item()

image1_path = ("/Users/metsa/PycharmProjects/pythonProject2/project/face_detecting/faces/pos/sean1.jpeg")
image2_path = ("//Users/metsa/PycharmProjects/pythonProject2/project/face_detecting/faces/neg/image7.jpeg")

image1 = load_and_preprocess_image(image1_path, mtcnn)
image2 = load_and_preprocess_image(image2_path, mtcnn)

if image1 is not None and image2 is not None:
    embedding_image1 = get_embedding(image1, resnet)
    embedding_image2 = get_embedding(image2, resnet)
    distance = calculate_l2_distance(embedding_image1, embedding_image2)
    print(f"Distance between two faces: {distance}")
else:
    print(f"The face could not detected!")

if (distance >= 0.98):
    print(f'The faces are different')
else:
    print(f'The faces are same')
