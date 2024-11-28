import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from utils.info import class_info


class LandmarkClassifier():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=None)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 188)  # 180은 클래스 개수입니다.
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load('./model/best_model_55.pth'))
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def inference(self, image: Image, top_k=5):

        if image.mode == 'RGBA':
            image = image.convert('RGB')



        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # 배치 차원을 추가합니다.

        # GPU를 사용할 경우 텐서를 GPU로 옮깁니다.
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            self.model.eval()  # 모델을 평가 모드로 설정합니다.
            output = self.model(input_batch)  # 모델을 통해 예측을 수행합니다.

        probs, indices = torch.topk(output, 5)
        probs = torch.nn.functional.softmax(probs, dim=1).cpu().numpy()[0]
        indices = indices.cpu().numpy()[0]

        indices = indices[:top_k+1]
        probs = probs[:top_k+1]

        print(probs)
        results = [{str(idx) : {'class_name': class_info[str(idx)], 'prob': float(prob)}} for idx, prob in zip(indices, probs)]
        
        return results