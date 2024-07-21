from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import cv2
import io
import base64
import torch
from langchain_core.messages import HumanMessage
import numpy as np
import time
from transformers import pipeline, AutoConfig
import datetime
#자체 모듈
from face_emo_module import read_emotion
from context_emotion_module import predict_emotion
# MPS 사용 가능 여부 확인 및 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 문장 임베딩 모델 로드
model = SentenceTransformer('jhgan/ko-sbert-sts')
model = model.to(device)

trained_model_path = "./trained_model"
toknizer_path = "gogamza/kobart-base-v2"
config = AutoConfig.from_pretrained(toknizer_path, num_labels=2)
nlg_pipeline = pipeline('text2text-generation', model=trained_model_path, tokenizer=toknizer_path, config=config, device='mps')

# 시각 정보 관련 예시 문장들
visual_examples = [
    "지금 보이는 것을 설명해줘",
    "현재 화면에 무엇이 있나요?",
    "카메라로 촬영한 이미지를 분석해줘",
    "눈앞의 장면을 묘사해주세요",
    "지금 보고 있는 것에 대해 말해줘",
    "이거 어때 보여?",
    # GPT 3.5
    "화면에 뭐가 보이는지 알려줘.",
    "이 사진에 대해서 설명해 줄 수 있어?",
    "이 이미지를 어떻게 이해해야 할까?",
    "내 앞에 있는 장면을 묘사해줘.",
    "카메라로 찍은 사진을 분석해 봐.",
    "지금 보고 있는 것에 대해 설명해 줄래?",
    "이 그림이 무엇을 나타내는지 말해줘.",
    "보이는 것을 자세히 설명해줘.",
    "지금 눈앞에 보이는 걸 묘사해줘.",
    "이 사진 속에 무엇이 있는지 말해 줄 수 있어?",
    "화면에 보이는 것을 설명해줘.",
    "현재 보고 있는 장면을 묘사해줄래?",
    "이 이미지를 해석해 줄 수 있어?",
    "이 그림에 대해 설명해줘.",
    "지금 화면에 무엇이 보이는지 말해줘.",
    "눈앞의 모습을 설명해줄래?",
    "카메라로 찍은 장면을 분석해줘.",
    "이 장면이 어떤지 말해줄래?",
    "이 이미지가 무엇을 나타내는지 설명해줘.",
    "현재 보이는 것에 대해 설명해줘.",
    # 클로드
    "보이는 걸 말해줘.",
    "이게 뭐로 보여?",
    "화면 설명해줘.",
    "이거 어떻게 생겼어?",
    "보이는 대로 말해봐.",
    "이 장면 묘사해줘.",
    "뭐가 보이니?",
    "이게 어떤 모양이야?",
    "화면에 뭐 있어?",
    "이거 설명 좀.",
    "보이는 거 설명해줘.",
    "이 모습 어때?",
    "뭐가 눈에 띄어?",
    "이거 어떻게 보여?",
    "화면 분석해줘.",
    "이거 뭐야?",
    "보이는 대로 얘기해줘.",
    "이 장면 어때?",
    "뭐가 보이는지 말해봐.",
    "이거 묘사해줘.",
    #내가 쓴거
    "내 헤드셋 어때?",
    "내꺼 어떄?",
    "내 모습 어때?",
    "이거 봐봐",
    "이것 좀 봐",
    "잘 봐봐",
    "봐바",
    "이거 쩐다"
]

style_map = {
    'formal': '문어체',
    'informal': '구어체',
    'android': '안드로이드',
    'azae': '아재',
    'chat': '채팅',
    'choding': '초등학생',
    'emoticon': '이모티콘',
    'enfp': 'enfp',
    'gentle': '신사',
    'halbae': '할아버지',
    'halmae': '할머니',
    'joongding': '중학생',
    'king': '왕',
    'naruto': '나루토',
    'seonbi': '선비',
    'sosim': '소심한',
    'translator': '번역기'
}

# 예시 문장들의 임베딩 계산
visual_embeddings = model.encode(visual_examples, convert_to_tensor=True)
visual_embeddings = visual_embeddings.to(device)

# LLaVA API 호출 함수 (가정)
def call_llava_api(image):
    # 이 부분은 실제 LLaVA API를 호출하는 코드로 대체해야 합니다.
    llava = Ollama(model="llava-phi3")
    response = llava("Objectively say only what is ", images=[image])
    return response

#말투 변환 llm 호출
def generate_text(text):
    print(f'\n내부 답변 : {text}\n')
    text = f"enfp 말투로 변환:{text}" # /======================================말투 변환======================================/
    out = nlg_pipeline(text, max_length=100)
    # print(out[0]['generated_text'])
    return out[0]['generated_text']

#텍스트 포맷 변경 함수
def format_output(text):
    return text.strip()


# 시각 정보 필요 여부 판단 함수
def needs_visual_info(prompt, threshold=0.51):
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    prompt_embedding = prompt_embedding.to(device)
    cosine_scores = util.pytorch_cos_sim(prompt_embedding, visual_embeddings)
    max_score = torch.max(cosine_scores)
    if max_score > threshold :
        print("needs_visual_info : ", prompt, " // ",max_score," // ","True")
    else :
        print("needs_visual_info : ", prompt, " // ",max_score," // ","False")
    return max_score > threshold

# 판단 및 이미지 처리 함수
def process_input(inputs):
    messages = inputs.to_messages()
    # messages = inputs
    human_contents = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
    print("process_input : ",human_contents)
    prompt = human_contents[-1] if human_contents else ""  # 리스트에서 문자열로 변경
    use_image = needs_visual_info(prompt)
    if use_image:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return [f"{prompt}\n이미지를 캡처할 수 없습니다."]
        
        # 카메라가 준비될 때까지 기다림
        for _ in range(30):  # 최대 30프레임 대기
            time.sleep(1)
            ret, frame = cap.read()
            if ret and frame is not None and not np.all(frame == 0):
                break
            time.sleep(0.1)
        
        if not ret or frame is None or np.all(frame == 0):
            print("유효한 프레임을 얻을 수 없습니다.")
            cap.release()
            return [f"{prompt}\n유효한 이미지를 캡처할 수 없습니다."]
        
        # 현재 시간을 가져옵니다
        current_time = datetime.datetime.now()
        # 시간을 원하는 형식으로 포맷팅합니다
        time_string = current_time.strftime("%Y%m%d_%H%M%S")

        cv2.imwrite(f"./captured/{time_string}.png", frame)
        print("이미지가 저장되었습니다: captured_image.png")
        cap.release()

        facial_description = read_emotion(ret=ret, frame=frame)
        text_predict = predict_emotion(prompt)

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_description = call_llava_api(img_str)
        print(f"appearance : {image_description}\n{facial_description}\ntone : {text_predict}\n\n{prompt} ")
        return [f"appearance : {image_description}\n{facial_description}\ntone : {text_predict}\n\n{prompt} "]
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return [f"{prompt}\n이미지를 캡처할 수 없습니다."]
        
        # 카메라가 준비될 때까지 기다림
        for _ in range(30):  # 최대 30프레임 대기
            time.sleep(1)
            ret, frame = cap.read()
            if ret and frame is not None and not np.all(frame == 0):
                break
            time.sleep(0.1)
        
        if not ret or frame is None or np.all(frame == 0):
            print("유효한 프레임을 얻을 수 없습니다.")
            cap.release()
            return [f"{prompt}\n유효한 이미지를 캡처할 수 없습니다."]
        
        # 현재 시간을 가져옵니다
        current_time = datetime.datetime.now()
        # 시간을 원하는 형식으로 포맷팅합니다
        time_string = current_time.strftime("%Y%m%d_%H%M%S")

        cv2.imwrite(f"./captured/{time_string}.png", frame)
        print("이미지가 저장되었습니다: captured_image.png")
        cap.release() 

        facial_description = read_emotion(ret=ret, frame=frame)
        text_predict = predict_emotion(prompt)

        print(f"\n{facial_description}\ntone : {text_predict}\n\nprompt : {prompt} ")
        return [f"\n{facial_description}\ntone : {text_predict}\n\nprompt : {prompt} "]

# Ollama LLM 초기화
llm = Ollama(model="Bllossom:8B")

#시각 능력 프롬프트
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI Vision Assistant.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
#발화 llm 프롬프트
prompt2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI friend, your name must be '지오퍼텐셜'. my name must be '화니' You answer in Korean or English. You understand the other person's facial expressions and tone of voice, and you consider the other person's feelings when answering. If the other person looks sad, angry, or fearful, be sure to tell them",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

#메인 체인
chain =  prompt | process_input | prompt2 | llm | StrOutputParser() | generate_text | format_output | StrOutputParser()
# chain =  prompt | process_input | prompt2 | llm | StrOutputParser()

# print(chain.invoke(["너 누구냐"]))