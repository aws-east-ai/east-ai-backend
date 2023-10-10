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

# import base64

app = FastAPI()


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


# 图片大小修改,并设置成 8 的倍数
def sd_resize_image(image: Image.Image, length=768):
    w, h = image.size
    # print("ori-size: ", (w,h))
    corp = (0, 0, w, h)
    if w > h:
        h = int((length * h / w))
        w = length
        ah = int(h / 8.0) * 8
        corp = (0, int((h - ah) / 2), w, ah + int((h - ah) / 2))
    elif w < h:
        w = int((length * w / h))
        h = length
        aw = int(w / 8.0) * 8
        corp = (int((w - aw) / 2), 0, int((w - aw) / 2) + aw, h)
    else:
        w = h = length
    rtn = image.resize((w, h), resample=Image.LANCZOS)
    if w % 8 != 0 or h % 8 != 0:
        rtn = rtn.crop(corp)
    return rtn


def get_str(item: dict, key, defaultValue: str | None = None):
    if key in item and item[key]:
        return item[key]
    return defaultValue


def get_int(item: dict, key, defaultValue: int | None = None):
    if key not in item:
        return defaultValue
    if not item[key]:
        return defaultValue
    try:
        return int(item[key])
    except:
        return defaultValue


ext_mimes = {
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


# 扩展名转化为 mime，只转化 图片类型
def get_mime_type(ext: str):
    # return ext_mimes[ext.lower()] or "application/octet-stream"
    return ext_mimes[ext.lower()] or None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
s3_bucket = os.environ.get("WORKSHOP_IMAGE_BUCKET", "east-ai-workshop")


llm_predictor = HuggingFacePredictor(endpoint_name="chatglm2-lmi-model")
pd_predictor = HuggingFacePredictor(endpoint_name="product-design-sd")
sam_predictor = HuggingFacePredictor(endpoint_name="grounded-sam")
inpaint_predictor = HuggingFacePredictor(endpoint_name="inpainting-sd")

# stream chat bot
s3 = boto3.resource("s3")
smr = boto3.client("sagemaker-runtime", region_name=region)
bedrock = boto3.client("bedrock-runtime", region_name=region)
parameters = {"max_length": 4092, "temperature": 0.01, "top_p": 0.8}
glm_entry_point = "chatglm2-lmi-model"

translate_client = boto3.client("translate")


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


@app.get("/")
def home():
    return "Hello world"


@app.post("/api/bedrock-sdxl")
def bedrock_sdxl(item: dict):
    prompt_res = translate_client.translate_text(
        Text=item["prompt"], SourceLanguageCode="auto", TargetLanguageCode="en"
    )
    prompt = "3D product render, {p}, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k".format(
        p=prompt_res["TranslatedText"]
    )

    negative_prompt = get_str(item, "negative_prompt")

    if negative_prompt:
        neg_prompt_res = translate_client.translate_text(
            Text=negative_prompt,
            SourceLanguageCode="auto",
            TargetLanguageCode="en",
        )
        negative_prompt = neg_prompt_res["TranslatedText"]

    steps = get_int(item, "steps", 30)
    # item["seed"] = int(item["seed"]) or -1
    height = get_int(item, "height", 512)
    width = get_int(item, "width", 512)
    # count = int(item["count"]) or 1
    style_preset = get_str(item, "style_preset", "3d-model")
    request = json.dumps(
        {
            "text_prompts": (
                [{"text": prompt, "weight": 1.0}]
                + [{"text": negative_prompt, "weight": -1.0}]
            ),
            "cfg_scale": 10,
            # "seed": -1,
            "steps": steps,
            "style_preset": style_preset,
            "width": width,
            "height": height,
        }
    )
    modelId = "stability.stable-diffusion-xl"
    # print(request)
    response = bedrock.invoke_model(body=request, modelId=modelId)
    response_body = json.loads(response.get("body").read())
    return {"images": [response_body["artifacts"][0].get("base64")]}
    # print(f"{base_64_img_str[0:80]}...")
    # image_1 = Image.open(io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8"))))
    # return image_1


@app.post("/api/product-design")
def product_design(item: dict):
    # 这里还需更细致的校验

    # 调用 translate 对 prompt 和 negative_prompt 进行翻译
    prompt_res = translate_client.translate_text(
        Text=item["prompt"], SourceLanguageCode="auto", TargetLanguageCode="en"
    )
    item[
        "prompt"
    ] = "3D product render, {p}, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k".format(
        p=prompt_res["TranslatedText"]
    )

    if item["negative_prompt"]:
        neg_prompt_res = translate_client.translate_text(
            Text=item["negative_prompt"],
            SourceLanguageCode="auto",
            TargetLanguageCode="en",
        )
        item["negative_prompt"] = neg_prompt_res["TranslatedText"]

    item["steps"] = int(item["steps"]) or 30
    item["seed"] = int(item["seed"]) or -1
    item["height"] = int(item["height"]) or 512
    item["width"] = int(item["width"]) or 512
    item["count"] = int(item["count"]) or 1
    item["output_image_dir"] = f"s3://{s3_bucket}/product-images/"

    # print(item)
    return pd_predictor.predict(item)


@app.post("/api/upload")
async def upload(file: UploadFile):
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents))
    except:
        return {"success": False, "message": "需要上传合法的图片。"}

    image = sd_resize_image(image, length=512)

    rnd_key = str(uuid.uuid4())
    now = datetime.now()
    year_str = now.strftime("%Y")
    day_str = now.strftime("%m%d")
    path_str = f"{year_str}{day_str}/"
    key_original = path_str + rnd_key + ".webp"

    buffer = BytesIO()
    image.save(buffer, format="WEBP")
    img_byte_arr = buffer.getvalue()

    s3.Bucket(s3_bucket).put_object(Key=f"images/{key_original}", Body=img_byte_arr)

    return {"success": True, "data": f"s3://{s3_bucket}/images/{key_original}"}


@app.post("/api/inpaint")
async def inpaint(item: dict):
    assert "input_image" in item
    assert "sam_prompt" in item
    output_mask_image_dir = f"s3://{s3_bucket}/mask-images/"

    sam_prompt_res = translate_client.translate_text(
        Text=item["sam_prompt"], SourceLanguageCode="auto", TargetLanguageCode="en"
    )

    mask_res = sam_predictor.predict(
        {
            "input_image": item["input_image"],
            "prompt": sam_prompt_res["TranslatedText"],
            "output_mask_image_dir": output_mask_image_dir,
        }
    )

    # 调用 translate 对 prompt 和 negative_prompt 进行翻译
    prompt_res = translate_client.translate_text(
        Text=item["prompt"], SourceLanguageCode="auto", TargetLanguageCode="en"
    )

    if item["negative_prompt"]:
        neg_prompt_res = translate_client.translate_text(
            Text=item["negative_prompt"],
            SourceLanguageCode="auto",
            TargetLanguageCode="en",
        )

    return inpaint_predictor.predict(
        {
            "prompt": prompt_res["TranslatedText"] or item["prompt"],
            "negative_prompt": neg_prompt_res["TranslatedText"]
            or item["negative_prompt"],
            "input_image": item["input_image"],
            "input_mask_image": mask_res["result"],
            "steps": int(item["steps"]) or 30,
            "sampler": item["sampler"],
            "seed": int(item["seed"]) or -1,
            "count": int(item["count"]) or 1,
        }
    )


@app.get("/api/s3-image/{s3_url:path}")
async def render_image_from_s3(s3_url: str):
    split_tup = os.path.splitext(s3_url)
    media_type = get_mime_type(split_tup[1])
    # 只允许特定的扩展名
    if not media_type:
        return None
    # 解析 bucket 和 key
    s3_urls = s3_url.replace("s3://", "").replace("s3:/", "").split("/")
    print("vvv|||", s3_url, s3_urls)
    n_bucket = s3_urls[0]
    key = s3_url.replace("s3://" + s3_bucket + "/", "").replace(
        "s3:/" + s3_bucket + "/", ""
    )

    xobject = s3.Object(n_bucket, key).get()
    img_bytes = xobject["Body"].read()
    image_stream = io.BytesIO(img_bytes)
    return StreamingResponse(content=image_stream, media_type=media_type)
