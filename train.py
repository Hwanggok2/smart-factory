import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. 데이터셋 경로 설정 (이 폴더 구조에 맞게 이미지를 넣어주세요)
# dataset/
# ├── train/
# │   ├── normal/
# │   └── defect/
# └── val/
#     ├── normal/
#     └── defect/
DATA_DIR = 'dataset'

def train_model(num_epochs=5, batch_size=8, learning_rate=0.0001):
    if not os.path.exists(DATA_DIR):
        print(f"'{DATA_DIR}' 폴더가 없습니다. 스크립트 상단의 주석을 참고하여 데이터셋 폴더를 생성해주세요.")
        # 사용자가 이미지를 넣기 편하도록 더미 폴더 구조 자동 생성
        os.makedirs(os.path.join(DATA_DIR, 'train', 'normal'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'train', 'defect'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'val', 'normal'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'val', 'defect'), exist_ok=True)
        print("더미 폴더 구조를 생성했습니다. 이미지를 넣고 다시 실행해주세요.")
        return

    # 2. 이미지 전처리 및 증강 (Data Augmentation)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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
        print("학습 데이터가 없습니다. 생성된 dataset/train/normal, defect 폴더에 이미지를 넣어주세요.")
        return

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"학습에 사용할 디바이스: {device}")

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"클래스 종류: {class_names} ({num_classes}개)")

    # 3. 사전 학습된 ViT 모델 로드 및 구조 변경
    print("사전 학습된 ViT 모델을 불러오는 중...")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    
    # 마지막 분류 층 수정 (우리의 클래스 개수에 맞게 조정)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # 손실 함수 및 최적화 알고리즘
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 학습 루프 진행
    print("학습 시작!")
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

    # 5. 최종 파인튜닝된 모델 가중치 저장
    torch.save({
        'class_names': class_names,
        'model_state_dict': model.state_dict(),
    }, 'fine_tuned_vit.pth')
    print("\n학습이 완료되어 'fine_tuned_vit.pth'로 저장되었습니다!")
    print("이제 백엔드 서버(main.py)를 재시작하면 자동으로 이 모델을 사용하게 됩니다.")

if __name__ == '__main__':
    train_model()
