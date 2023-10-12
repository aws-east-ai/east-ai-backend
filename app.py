from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sagemaker.huggingface.model import HuggingFacePredictor
from fastapi.responses import StreamingResponse
from fastapi import UploadFile
from PIL import Image
import io
from io import BytesIO
import boto3
from datetime import datetime
import uuid
import json
import os
from utils.image import sd_resize_image
from utils.common import get_int, get_str

# import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Router mapping
from controller.home import home_router
from controller.bedrock import bedrock_router
from controller.paint import paint_router
from controller.tools import tools_router

app.include_router(home_router)
app.include_router(bedrock_router)
app.include_router(paint_router)
app.include_router(tools_router)


region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
# s3_bucket = os.environ.get("WORKSHOP_IMAGE_BUCKET", "east-ai-workshop")

llm_predictor = HuggingFacePredictor(endpoint_name="chatglm2-lmi-model")
# pd_predictor = HuggingFacePredictor(endpoint_name="product-design-sd")
# sam_predictor = HuggingFacePredictor(endpoint_name="grounded-sam")
# inpaint_predictor = HuggingFacePredictor(endpoint_name="inpainting-sd")

# stream chat bot
s3 = boto3.resource("s3")
smr = boto3.client("sagemaker-runtime", region_name=region)
parameters = {"max_length": 4092, "temperature": 0.01, "top_p": 0.8}
glm_entry_point = "chatglm2-lmi-model"


# 解析 stream
class StreamScanner:
    def __init__(self):
        self.buff = io.BytesIO()
        self.read_pos = 0

    def write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)

    def readlines(self):
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            if line[-1] != b"\n":
                self.read_pos += len(line)
                yield line[:-1]

    def reset(self):
        self.read_pos = 0


@app.websocket("/api/chat-bot")
async def chat_bot(websocket: WebSocket):
    # websocket.on_disconnect()
    # await manager.connect(websocket)
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            # TODO: error handle，如果不是 json 格式则会报错
            item = json.loads(data)
            history = item["history"] if "history" in item else []
            pattern = item["pattern"]
            prompt_pattern = patterns[pattern]
            prompt = item["prompt"]
            prompt = prompt if history else prompt_pattern + "\n\n" + prompt

            question = {"status": "begin", "question": prompt}
            await websocket.send_text(json.dumps(question, ensure_ascii=False))

            response_model = smr.invoke_endpoint_with_response_stream(
                EndpointName=glm_entry_point,
                Body=json.dumps(
                    {"inputs": prompt, "parameters": parameters, "history": history}
                ),
                ContentType="application/json",
            )

            event_stream = response_model["Body"]
            scanner = StreamScanner()
            resp = {}
            for event in event_stream:
                scanner.write(event["PayloadPart"]["Bytes"])
                for line in scanner.readlines():
                    try:
                        resp = json.loads(line)["outputs"]
                        await websocket.send_text(resp["outputs"])
                    except Exception as e:
                        continue
            result_end = (
                '{"status": "done", "history": '
                + json.dumps(resp["history"], ensure_ascii=False)
                + "}"
            )
            await websocket.send_text(result_end)
    except WebSocketDisconnect:
        print(f"Client left")


patterns = {
    "redbook": "你是一个时尚的年轻人，喜欢用emoji，请根据下面的内容写一段小红书的种草文案: ",
    "zhihu": "你是一个知识博学的学者，请根据下面的内容写一段文章，发表在知乎上: ",
    "weibo": "请根据下面的内容写一段微博的短文，140 字以内: ",
    "gongzhonghao": "你是一名思想者，请根据下面的内容写一段公众号的文章: ",
    "dianping": "你是一产品使用者，请根据下面的内容写一段点评的评论: ",
    "toutiao": "你是一个记者，请根据下面的内容写一则头条的新闻: ",
    "zhidemai": "你是一个经验丰富的导购，请根据下面的内容写一段值得买的文章: ",
    "douyin": "你是一个短片导演，请根据下面的内容写一段抖音的拍摄剧本: ",
    "kuaishou": "你是一个短片导演，请根据下面的内容写一段快手的短片剧本: ",
}


@app.post("/api/write-marketing-text")
def write_marketing_text(item: dict):
    history = item["history"] if "history" in item else []
    pattern = item["pattern"]
    prompt_pattern = patterns[pattern]
    prompt = item["prompt"]
    prompt = prompt if history else prompt_pattern + "\n\n" + prompt
    res = llm_predictor.predict({"inputs": prompt, "parameters": {"history": history}})
    return res
