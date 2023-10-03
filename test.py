import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

class CNN(nn.Module):                         #nn.Module을 상속받아 CNN 클래스를 정의합니다. 이 클래스는 PyTorch 모델을 나타내며, 신경망 구조와 forward 연산을 정의합니다.
    def __init__(self):                       #클래스의 생성자 메서드를 정의합니다.
        super(CNN, self).__init__()           #부모 클래스인 nn.Module의 생성자를 호출하여 초기화합니다.
        self.conv1 = nn.Conv2d(               #첫 번째 합성곱 층을 정의합니다.
            in_channels=3,                          #in_channels는 입력 채널 수로 RGB 이미지의 경우 3입니다.
            out_channels=8,                         #out_channels는 출력 채널 수로 8로 설정하고,
            kernel_size=3,                          #커널 크기는 3x3이며,
            padding=1)                              #패딩을 1로 설정합니다
        self.conv2 = nn.Conv2d(               #두 번째 합성곱 층을 정의합니다.
            in_channels=8,                          #입력 채널 수는 이전 층의 출력 채널 수인 8로 설정하고, 출력 채널 수는 16로 설정하며, 커널 크기는 3x3이며, 패딩을 1로 설정합니다.
            out_channels=16,                        #출력 채널 수는 16로 설정하며,
            kernel_size=3,                          #커널 크기는 3x3이며,
            padding=1)                              #패딩을 1로 설정합니다.
        self.pool = nn.MaxPool2d(             #대 풀링(Max Pooling) 층을 정의합니다. 커널 크기와 스트라이드를 모두 2로 설정하여 이미지를 축소합니다.
            kernel_size=2,
            stride=2
        )
        self.fc1 = nn.Linear(32 * 32 * 16, 128)     #완전 연결(fully connected) 층을 정의합니다. 입력 크기는 이전 층의 출력 크기인 32x32x16로 설정하고, 출력 크기는 128로 설정합니다.
        self.fc2 = nn.Linear(128, 64)               #두 번째 완전 연결 층을 정의합니다. 입력 크기는 이전 층의 출력 크기인 128로 설정하고, 출력 크기는 64로 설정합니다.
        self.fc3 = nn.Linear(64, 10)                #세 번째 완전 연결 층을 정의합니다. 입력 크기는 이전 층의 출력 크기인 64로 설정하고, 출력 크기는 클래스의 개수에 해당하는 10으로 설정합니다.

    def forward(self, x):                     #모델의 forward 연산을 정의하는 메서드입니다.
        x = self.conv1(x)                       #입력 데이터에 첫 번째 합성곱 층을 적용합니다.
        x = torch.relu(x)                       #ReLU 활성화 함수를 적용합니다.
        x = self.pool(x)                        #최대 풀링 층을 적용하여 데이터를 축소합니다.
        x = self.conv2(x)                       #두 번째 합성곱 층을 적용합니다.
        x = torch.relu(x)                       #ReLU 활성화 함수를 적용합니다.
        x = self.pool(x)                        #최대 풀링 층을 적용하여 데이터를 다시 축소합니다.

        x = x.view(-1, 32 * 32 * 16)            #데이터를 평탄화(flatten)하여 완전 연결 층에 입력할 수 있는 형태로 변환합니다.
        x = self.fc1(x)                         #첫 번째 완전 연결 층을 적용합니다.
        x = torch.relu(x)                       #ReLU 활성화 함수를 적용합니다.
        x = self.fc2(x)                         #두 번째 완전 연결 층을 적용합니다.
        x = torch.relu(x)                       #ReLU 활성화 함수를 적용합니다.
        x = self.fc3(x)                         #세 번째 완전 연결 층을 적용합니다.
        x = torch.log_softmax(x, dim=1)         #출력을 로그 소프트맥스 함수를 통과시켜 클래스 확률 분포를 얻습니다.
        return x

# 1. 모델 정의 및 미리 학습된 가중치 로드
model = CNN()  # 이전에 정의한 모델을 생성합니다.
model.load_state_dict(torch.load('model.pt'))  # 'model.pt' 파일로부터 미리 학습된 모델 가중치를 로드합니다.
model.eval()  # 모델을 평가(추론) 모드로 설정합니다.

# 2. 추가한 이미지를 로드하고 전처리
image_path = '44079668_34dfee3da1_n.jpg'
image = Image.open(image_path)


preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = preprocess(image).unsqueeze(0)

# 3. 모델에 이미지 입력하여 예측 수행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_image = input_image.to(device)

with torch.no_grad():
    output = model(input_image)

    # 소프트맥스 함수를 적용하여 확률 분포를 얻습니다.
    probabilities = torch.softmax(output, dim=1)

    class_probabilities = probabilities[0] * 100  # 확률값을 백분율로 변환


_, predicted_class = output.max(1)
predicted_class = predicted_class.item()

# 4. 예측 결과 출력
for i, prob in enumerate(class_probabilities):
    print(f"Class {i}: {prob:.2f}%")


print("Predicted class:", predicted_class)