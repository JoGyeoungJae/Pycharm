from fastapi import FastAPI, UploadFile
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import os
import uuid

app = FastAPI()
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1=nn.Conv2d(  #128*128
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        padding=1
    )
    self.conv2=nn.Conv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1
    )
    self.conv3=nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1
    )
    self.conv4=nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        padding=1
    )
    self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1=nn.Linear(8*8*64, 128)
    self.fc2=nn.Linear(128, 64)
    self.fc3=nn.Linear(64, 5)

  def forward(self, x): #[3, 128, 128]
    x=self.conv1(x)
    x=torch.relu(x)
    x=self.pool(x)  #[8, 64, 64]
    x=self.conv2(x)
    x=torch.relu(x)
    x=self.pool(x) #[16, 32,32]
    x=self.conv3(x)
    x=torch.relu(x)
    x=self.pool(x) #[32, 16,16]
    x=self.conv4(x)
    x=torch.relu(x)
    x=self.pool(x)  #[64, 8, 8]

    x=x.view(-1, 8*8*64)
    x=self.fc1(x)
    x=self.fc2(x)
    x=self.fc3(x)
    x=torch.log_softmax(x, dim=1)
    return x






@app.post("/upload")
async def upload_image(image: UploadFile):

    save_path = "C:/image/land/"
    with open(save_path + image.filename, "wb") as img_file:
        img_file.write(image.file.read())

    model = CNN()
    model.load_state_dict(torch.load('model_test.pt', map_location=torch.device('cpu')))
    model.eval()


    real_image_path = os.path.join(save_path, image.filename)
    image_obj = Image.open(real_image_path)


    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = preprocess(image_obj).unsqueeze(0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_image = input_image.to(device)


    with torch.no_grad():
        output = model(input_image)
        probabilities = torch.softmax(output, dim=1)
        class_probabilities = probabilities[0] * 100
    _, predicted_class = output.max(1)
    predicted_class = predicted_class.item()

    percent = 0  # 초기값을 0으로 설정
    predicted_class = -1  # 초기값을 -1로 설정 (클래스 인덱스 저장용)

    # 클래스 확률 출력 및 최대 확률 계산
    for i, prob in enumerate(class_probabilities):
        print(f"Class {i}: {prob:.2f}%")
        if prob > percent:  # 현재 클래스 확률이 최대 확률보다 크다면
            percent = prob  # 최대 확률을 업데이트
            predicted_class = i  # 예측 클래스 인덱스 업데이트
    print("maxpercent",percent)
    print("Predicted class:", predicted_class)

    # 디렉터리 내의 폴더 리스트 가져오기
    folder_list = [folder for folder in os.listdir(save_path) if
                   os.path.isdir(os.path.join(save_path, folder))]

    # 결과 이미지 저장 경로 생성
    result_save_dir = os.path.join(save_path, str(folder_list[predicted_class]))
    os.makedirs(result_save_dir, exist_ok=True)  # 결과 폴더가 없으면 생성



    # UUID 생성
    unique_filename = str(uuid.uuid4())

    # 확장자 유지 또는 변경
    file_extension = image.filename.split(".")[-1]  # 이미지 파일 확장자 추출
    unique_filename_with_extension = f"{unique_filename}.{file_extension}"

    # 결과 이미지 저장 경로 생성
    result_image_path = os.path.join(result_save_dir, unique_filename_with_extension)

    # 이미지 파일 저장 (이름이 UUID로 변경됨)
    os.rename(real_image_path, result_image_path)
    return predicted_class

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
