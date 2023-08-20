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
def productDesign():
    pass
