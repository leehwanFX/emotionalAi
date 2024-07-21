from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
llm = Ollama(model="Bllossom:q8_0")

# from langchain_community.llms import Ollama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from transformers import pipeline, AutoConfig
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.messages import AIMessage, HumanMessage
# from ollama_function import get_capture
# from langchain_experimental.llms.ollama_functions import OllamaFunctions
# from langchain_core.tools import tool
# import cv2
# from PIL import Image
# import io
# import base64
# from langchain.llms import Ollama

# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
# ollama_llm = OllamaFunctions(model="Bllossom:q8_0")

# model_path = "../../../BART/final_model"
# model_name = "gogamza/kobart-base-v2"
# config = AutoConfig.from_pretrained(model_name, num_labels=2)
# nlg_pipeline = pipeline('text2text-generation', model=model_path, tokenizer=model_name, config=config, device='mps')

# prompt1 = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful AI Vision Assistant.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# prompt2 = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             ""
#         )
#     ]
# )

# def generate_text(text):
#     text = f"enfp 말투로 변환:{text}"
#     out = nlg_pipeline(text, max_length=60)
#     return out[0]['generated_text']

# def format_output(text):
#     return text.strip()

# # ====
# def capture_and_analyze_image() -> str:
#     """눈 앞에 보이는 것을 분석해줍니다."""
#     # 카메라 캡처
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     cap.release()

#     if not ret:
#         return "카메라에서 이미지를 캡처하는 데 실패했습니다."

#     # 캡처한 이미지 저장
#     cv2.imwrite('./captured/captured_image.jpg', frame)
#     print("이미지가 저장되었습니다: captured_image.jpg")

#     # OpenCV 이미지를 PIL 이미지로 변환
#     img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # 이미지를 base64로 인코딩
#     buffered = io.BytesIO()
#     img_pil.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()

#     # llava 모델 초기화 및 이미지 분석
#     llava = Ollama(model="llava-phi3")
#     response = llava("Objectively analyze this image as you see it", images=[img_str])

#     return response

# # LangChain LLM 초기화 및 도구 바인딩
# llm = OllamaFunctions(model="Bllossom:q8_0", format="json")
# llm_with_tools = llm.bind_tools([capture_and_analyze_image])


# from langchain_core.messages import HumanMessage, ToolMessage

# def get_capture(prompt):
#     """눈 앞에 보이는 것을 설명해줍니다."""
#     print("function called")
#     query = prompt
#     messages = [HumanMessage(query)]
#     ai_msg = llm_with_tools.invoke(messages)
#     messages.append(ai_msg)
#     for tool_call in ai_msg.tool_calls:
#         selected_tool = {"capture_and_analyze_image": capture_and_analyze_image}[tool_call["name"].lower()]
#         tool_output = selected_tool.invoke(tool_call["args"])
#         messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
#     return str(messages[2].content)
# # ====


# # chain = prompt | llm | StrOutputParser() | generate_text | format_output | StrOutputParser()
# chain1 = prompt1 | get_capture | ollama_llm | StrOutputParser() | prompt2
# print(chain1.invoke("눈 앞에 보이는 것을 분석해주세요"))

# response = chain.invoke({"messages": [{"role": "human", "content": "안녕하세요, 오늘 날씨가 어때요?"}]})
# response1 = chain1.invoke({"messages": [{"role": "human", "content": "안녕하세요, 오늘 날씨가 어때요?"}]})
# print(response)
# print(response1)

