import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, test_loader, topk=(1,)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    maxk = max(topk)  # top-k에서 k의 최댓값을 가져옴
    correct_topk = {k: 0 for k in topk}  # 각 top-k 별로 맞춘 개수를 저장할 딕셔너리

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # topk의 예측값과 인덱스를 가져옴
            _, predicted_topk = outputs.topk(maxk, 1, True, True)
            
            for k in topk:
                predicted_k = predicted_topk[:, :k]
                correct_k = predicted_k.eq(labels.view(-1, 1).expand_as(predicted_k))
                correct_topk[k] += correct_k.view(-1).float().sum(0, keepdim=True).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print('Accuracy of the model on the test images: {}%'.format(100 * correct / total))
    for k, correct_k in correct_topk.items():
        print(f"Top-{k} Accuracy: {100 * correct_k / total}%")
    print('Precision:', precision_score(all_labels, all_predictions, average='macro'))
    print('Recall:', recall_score(all_labels, all_predictions, average='macro'))
    print('F1 Score:', f1_score(all_labels, all_predictions, average='macro'))

def main():
    # 모델 정의 및 가중치 로드
    model = models.model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 188)
    model.load_state_dict(torch.load('/home/yongjang/projects/LMClassification/LandmarkClassification/model/best_model_55.pth'))

    # 데이터 전처리 정의
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 테스트 데이터셋 로드
    test_dataset = torchvision.datasets.ImageFolder(root='/home/datasets/LMImages/val_data', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 모델 평가
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()