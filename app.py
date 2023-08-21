from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sagemaker.huggingface.model import HuggingFacePredictor

# from dotenv import load_dotenv
# import git
# import os
# import subprocess

# load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# src_dir = os.getenv("SRC_DIR")
# git_username = os.getenv("GIT_USERNAME")
# git_token = os.getenv("GIT_TOKEN")


llm_predictor = HuggingFacePredictor(
    endpoint_name="chatglm2-lmi-model-2023-08-16-08-23-01-220"
)

pd_predictor = HuggingFacePredictor(
    endpoint_name="product-design-sd-2023-08-16-08-18-44-842"
)

patterns = {
    "redbook": "请根据下面的内容写一段小红书的种草文案: ",
    "zhihu": "请根据下面的内容写一段知乎的文章: ",
    "weibo": "请根据下面的内容写一段微博的文章: ",
    "gongzhonghao": "请根据下面的内容写一段公众号的文章: ",
    "dianping": "请根据下面的内容写一段点评的文章: ",
    "toutiao": "请根据下面的内容写一段头条的文章: ",
    "zhidemai": "请根据下面的内容写一段值得买的文章: ",
    "douyin": "请根据下面的内容写一段抖音的拍摄文案: ",
    "kuaishou": "请根据下面的内容写一段快手的拍摄文案: ",
}


@app.post("/api/write-marketing-text")
def write_marketing_text(item: dict):
    history = item["history"] if "history" in item else []
    pattern = item["pattern"]
    prompt_pattern = patterns[pattern]
    prompt = item["prompt"]
    prompt = prompt if history else prompt_pattern + "\n\n" + prompt
    return llm_predictor.predict({"inputs": prompt, "parameters": {"history": history}})


@app.get("/")
def home():
    return "Hello world"


@app.post("/api/upload")
def upload():
    pass


@app.post("/api/product-design")
def productDesign(item: dict):
    print(item)

    # 这里还需更细致的校验
    item["steps"] = int(item["steps"])
    item["seed"] = int(item["seed"])
    item["height"] = int(item["height"])
    item["width"] = int(item["width"])
    item["count"] = int(item["count"])

    # item["steps"] = int(item["steps"])
    # inputs = {
    #     "prompt": "3D product render, futuristic armchair, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k",
    #     "negative_prompt": "EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), cropped, text, jpeg artifacts, signature, watermark, username, sketch, cartoon, drawing, anime, duplicate, blurry, semi-realistic, out of frame, ugly, deformed",
    #     "steps": 30,
    #     "sampler": "dpm2_a",
    #     "seed": -1,
    #     "height": 512,
    #     "width": 512,
    #     "count": 1,
    # }
    return pd_predictor.predict(item)
