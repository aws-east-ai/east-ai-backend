from sagemaker import Predictor, serializers, deserializers
from utils.painter import product_design, inpaint
from .aws import translate


agent_predictor = Predictor(
    endpoint_name="chatglm3-lmi-model",
    serializer=serializers.JSONSerializer(),
    deserializer=deserializers.JSONDeserializer(),
)

tools = [
    {
        "name": "image-generator",
        "description": "根据提示词生成相应的图片",
        "parameters": {
            "type": "object",
            "properties": {"prompts": {"description": "用于描述图片的提示词"}},
            "required": ["prompts"],
        },
    },
    {
        "name": "change-background",
        "description": "输入的图片url，根据提示词，替换图片中指定对象后的背景",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"description": "图片所在的url地址"},
                "object": {"description": "图片中的对象"},
                "prompts": {"description": "需要生成的背景提示词"},
            },
            "required": ["url", "object", "prompts"],
        },
    },
]

system_info = {
    "role": "system",
    "content": "Answer the following questions as best as you can. You have access to the following tools:",
    "tools": tools,
}
history = [system_info]


def parse_task(query):
    # query = "请根据以下提示词生成图片：现代风帐篷"
    # query = "请将以下图片（https://awsiot.top/image.png）中人的背景替换为雪山"
    res = agent_predictor.predict({"inputs": query, "parameters": {"history": history}})
    return res["response"]
