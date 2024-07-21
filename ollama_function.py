import cv2
from PIL import Image
import io
import base64
from langchain_community.llms import Ollama
from langchain.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions

def capture_and_analyze_image() -> str:
    """눈 앞에 보이는 것을 분석해줍니다."""
    # 카메라 캡처
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "카메라에서 이미지를 캡처하는 데 실패했습니다."

    # 캡처한 이미지 저장
    cv2.imwrite('./captured/captured_image.jpg', frame)
    print("이미지가 저장되었습니다: captured_image.jpg")

    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 이미지를 base64로 인코딩
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # llava 모델 초기화 및 이미지 분석
    llava = Ollama(model="llava-phi3")
    response = llava("Objectively analyze this image as you see it", images=[img_str])

    return response

# LangChain LLM 초기화 및 도구 바인딩
llm = OllamaFunctions(model="Bllossom:q8_0", format="json")
llm_with_tools = llm.bind_tools([capture_and_analyze_image])


from langchain_core.messages import HumanMessage, ToolMessage

def get_capture(query):
    messages = [HumanMessage(query)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"capture_and_analyze_image": capture_and_analyze_image}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    return messages[2].content