from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import boto3
import os
from utils.bot import claude2_bot, claude3_bot


ws_router = APIRouter()
smr = boto3.client("sagemaker-runtime")
bedrock = boto3.client(service_name="bedrock-runtime")


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
    "twitter": "Please write a short post on Twitter based on the following tips: ",
    "instagram": "You are a fashion influencer and love to use emojis, please write a post on Instagram based on the following tips:",
    "tiktok": "You are a stylish short film director, please write a TikTok shooting script according to the prompts: ",
    "youtube": "You are a Youtuber, please write a shooting script according to the prompts: ",
    "medium": "You are a knowledgeable writer and scientist, please follow the tips below to write a long essay, either beautifully written, or with technical depth:",
    "freestyle": "",
}


@ws_router.websocket("/api/chat-bot")
async def chat_bot(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            # TODO: error handle，如果不是 json 格式则会报错
            # print(data)
            item = json.loads(data)
            model_id = item["model_id"] if "model_id" in item else "chatglm2"

            history = item["history"] if "history" in item else []

            pattern = item["pattern"]
            prompt_pattern = patterns[pattern]
            prompt = item["prompt"]
            prompt = prompt if history else prompt_pattern + "\n\n" + prompt
            question = {"status": "begin", "question": prompt}
            await websocket.send_text(json.dumps(question, ensure_ascii=False))

            if model_id == "chatglm2":
                await ask_chatglm2(websocket, prompt, history)
            if model_id == "chatglm3":
                await ask_chatglm3(websocket, prompt, history)
            elif model_id == "bedrock_claude2":
                # await ask_bedrock_claude2(websocket, prompt, history)
                await claude2_bot(bedrock, websocket, prompt, history)
            elif model_id == "bedrock_claude3":
                # await ask_bedrock_claude2(websocket, prompt, history)
                await claude3_bot(bedrock, websocket, prompt, history)
    except WebSocketDisconnect:
        print(f"Client left")


async def ask_chatglm3(websocket: WebSocket, prompt: str, history):
    parameters = {"max_length": 4092, "temperature": 0.01, "top_p": 0.8}

    response_model = smr.invoke_endpoint_with_response_stream(
        EndpointName="chatglm3-lmi-model",
        Body=json.dumps(
            {"inputs": prompt, "parameters": parameters, "history": history}
        ),
        ContentType="application/json",
    )
    stream = response_model.get("Body")

    if stream:
        chunk_str_full = ""
        for event in stream:
            chunk = event.get("PayloadPart")
            # print(chunk)
            # print("------------------------------------------")
            if chunk:
                chunk_now = chunk.get("Bytes").decode()
                # print(chunk_now)
                # print("------------------------------------------")
                chunk_str_full = chunk_str_full + chunk_now
                if chunk_str_full.strip().endswith(
                    "]}}"
                ) and chunk_str_full.strip().startswith("{"):
                    # print(chunk_str_full)
                    chunk_obj = json.loads(chunk_str_full)
                    # print(chunk_obj)
                    result = chunk_obj["outputs"]["outputs"]
                    # result = chunk_obj["response"]
                    chunk_str_full = ""
                    await websocket.send_text(result)

    result_end = '{"status": "done"}'
    await websocket.send_text(result_end)


async def ask_chatglm2(websocket: WebSocket, prompt: str, history):
    parameters = {"max_length": 4092, "temperature": 0.01, "top_p": 0.8}

    response_model = smr.invoke_endpoint_with_response_stream(
        EndpointName="chatglm2-lmi-model",
        Body=json.dumps(
            {"inputs": prompt, "parameters": parameters, "history": history}
        ),
        ContentType="application/json",
    )
    stream = response_model.get("Body")
    if stream:
        chunk_str_full = ""
        for event in stream:
            chunk = event.get("PayloadPart")
            if chunk:
                chunk_str_full = chunk_str_full + chunk.get("Bytes").decode()
                # print(chunk_str_full)
                if chunk_str_full.strip().endswith(
                    "]}}"
                ) and chunk_str_full.strip().startswith("{"):
                    chunk_obj = json.loads(chunk_str_full)
                    result = chunk_obj["outputs"]["outputs"]
                    chunk_str_full = ""
                    await websocket.send_text(result)

    result_end = '{"status": "done"}'
    await websocket.send_text(result_end)
