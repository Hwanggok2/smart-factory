import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 데이터셋 경로 (배터리 CT 스캔용 정상/비정상 데이터가 저장된 곳)
DATA_DIR = 'ct_dataset'

def train_resnet_model(num_epochs=20, batch_size=16, learning_rate=0.0001):
    if not os.path.exists(DATA_DIR):
        print(f"'{DATA_DIR}' 폴더가 없습니다. 폴더를 생성하고 배터리 CT 이미지를 넣어주세요.")
        os.makedirs(os.path.join(DATA_DIR, 'train', 'normal'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'train', 'defect'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'val', 'normal'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'val', 'defect'), exist_ok=True)
        print("더미 폴더 구조를 생성했습니다. 이미지를 넣고 다시 실행해주세요.")
        return

    # 이미지 전처리 및 증강 (Data Augmentation) - CT 이미지에 맞게 최적화
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), # CT는 수직 반전도 의미가 있음
            transforms.RandomRotation(15),   # 미세한 각도 조절
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                          for x in ['train', 'val']}
    except FileNotFoundError:
        print("데이터셋 폴더의 구조가 올바르지 않습니다.")
        return
        
    if len(image_datasets['train']) == 0:
        print("학습 데이터가 없습니다. 생성된 dataset/train 폴더에 이미지를 넣어주세요.")
        return

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"학습에 사용할 디바이스: {device}")

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"클래스 종류: {class_names} ({num_classes}개)")

    # 사전 학습된 ResNet50 모델 로드
    print("사전 학습된 ResNet50 모델을 불러오는 중...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # 마지막 분류 층 수정 (FC 층)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # 손실 함수 및 최적화 알고리즘
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프 진행
    print("ResNet50 학습 시작!")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 최종 파인튜닝된 ResNet 모델 가중치 저장
    torch.save({
        'class_names': class_names,
        'model_state_dict': model.state_dict(),
    }, 'fine_tuned_resnet_battery.pth')
    print("\n학습이 완료되어 'fine_tuned_resnet_battery.pth'로 저장되었습니다!")

if __name__ == '__main__':
    train_resnet_model()
