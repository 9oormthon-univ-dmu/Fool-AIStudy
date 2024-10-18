# 9 Week

## 9주차  

### 강의 키워드: EfficientNet

### 강의 내용

- **EfficientNet**:
- torchvision.models를 사용한 사전 훈련된 모델 활용
- EfficientNet 모델 시리즈 사용 (EfficientNetB2)

'def create_effnetb2_model(num_classes:int=3, seed:int=42):
    # EfficientNetB2 모델 생성
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)
    
    # 기존 모델 레이어 동결
    for param in model.parameters():
        param.requires_grad = False
        
    # 새로운 분류기 추가
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )
    
    return model'

- EfficientNet의 특징:
1. 효율적인 모델 스케일링 방법 사용
2. 이미지 분류 작업에서 높은 성능 달성
3. 다양한 크기의 모델 제공 (B0부터 B7까지)

- 전이 학습 과정:
1. 사전 훈련된 EfficientNetB2 모델 로드
2. 기존 모델의 가중치 동결
3. 새로운 분류기 레이어 추가
4. 새로운 데이터셋에 대해 미세 조정 (fine-tuning)

- EfficientNet 사용의 이점:
1. 높은 정확도와 효율성
2. 적은 파라미터로 좋은 성능 달성
3. 다양한 크기의 모델 중 선택 가능

- **FocalLoss**:
- 불균형한 데이터셋에서 분류 문제를 다룰 때 유용한 손실 함수
- 쉬운 예제(잘 분류된 예제)의 영향을 줄이고 어려운 예제에 더 집중하도록 설계됨

- FocalLoss의 특징:
1. alpha: 클래스 불균형을 조정하는 가중치 팩터
2. gamma: 잘 분류된 예제의 영향을 줄이는 focusing 파라미터
3. 기존의 CrossEntropyLoss를 기반으로 하되, 어려운 예제에 더 큰 가중치를 부여

- FocalLoss 사용 예:
'# FocalLoss 인스턴스 생성
criterion = FocalLoss(alpha=1, gamma=2)

# 훈련 루프 내에서 사용
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # 역전파 및 최적화 단계'

- FocalLoss의 장점:
1. 클래스 불균형 문제 해결에 효과적
2. 어려운 예제에 대한 학습을 강화
3. 객체 검출 등의 작업에서 성능 향상
