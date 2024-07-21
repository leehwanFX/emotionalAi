import joblib
import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# BERT 모델 로드
class BertFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')
        self.bert.eval()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1)

# 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BertFeatureExtractor().to(device)

# 특징 추출 함수
def get_embeddings(texts, model, device):
    model.eval()
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=32)
    
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="어조 분석 중", unit="batch"):
            input_ids, attention_mask = [t.to(device) for t in batch]
            embeddings = model(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0).numpy()

# 저장된 파이프라인 모델 불러오기
pipeline = joblib.load('./naive_bayes_model/naive_bayes_pipeline.joblib')

# 레이블 인코더 불러오기 (이전에 저장했다고 가정)
label_encoder = joblib.load('./naive_bayes_model/label_encoder.joblib')

# 테스트 함수
def predict_emotion(text):
    # 텍스트를 임베딩으로 변환
    print("\n")
    embedding = get_embeddings([text], model, device)
    # 감정 예측
    prediction = pipeline.predict(embedding)
    # 예측된 레이블을 원래 감정으로 변환
    predicted_emotion = label_encoder.inverse_transform(prediction)[0]
    return predicted_emotion


# print(predict_emotion("앱등이들 대거 몰렸네."))