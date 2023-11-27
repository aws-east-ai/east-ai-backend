from sagemaker import Predictor, serializers, deserializers
from utils.painter import product_design, inpaint
from .aws import translate


predictor = Predictor(
    endpoint_name="chatglm3-lmi-model",
    serializer=serializers.JSONSerializer(),
    deserializer=deserializers.JSONDeserializer(),
)

tools = [
    {
        "name": "iamge-generator",
        "description": "根据提示词生成相应的图片",
        "parameters": {
            "type": "object",
            "properties": {
                "prompts": {
                    "description": "用于描述图片的提示词"
                }
            },
            "required": ['prompts']
        }
    },
    {
        "name": "change-background",
        "description": "输入的图片url，根据提示词，替换图片中指定对象后的背景",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "description": "图片所在的url地址"
                },
                "object": {
                    "description": "图片中的对象"
                },
                "prompts": {
                    "description": "需要生成的背景提示词"
                }
            },
            "required": ['url', 'object', 'prompts']
        }
    }
]
system_info = {"role": "system",
               "content": "Answer the following questions as best as you can. You have access to the following tools:", "tools": tools}
history = [system_info]


def agent_run(query):
    # query = "请根据以下提示词生成图片：现代风帐篷"
    # query = "请将以下图片（https://awsiot.top/image.png）中人的背景替换为雪山"
    res = predictor.predict(
        {"inputs": query, "parameters": {"history": history}}
    )
    # print(res['response'])
    # {'name': 'iamge-generator', 'parameters': {'prompts': '现代风帐篷'}}
    # {'name': 'change-background', 'parameters': {'url': 'https://awsiot.top/image.png', 'object': 'person', 'prompts': 'snow mountain'}}
    if res['response']['name'] == 'iamge-generator':
        prompts = translate(res['response']['parameters']['prompts'])
        item = {
            "prompt": "3D product render, {}, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k".format(prompts),
            "negative_prompt": "EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), cropped, text, jpeg artifacts, signature, watermark, username, sketch, cartoon, drawing, anime, duplicate, blurry, semi-realistic, out of frame, ugly, deformed",
            "steps": 30,
            "sampler": "ddim",
            "seed": -1,
            "height": 512,
            "width": 512,
            "count": 1,
        }
        return product_design(item)
    elif res['response']['name'] == 'change-background':
        input_image = res['response']['parameters']['url']
        sam_prompt = res['response']['parameters']['object']
        inpaint_prompt = res['response']['parameters']['prompts']
        item = {
            "input_image": input_image,
            "sam_prompt": sam_prompt,
            "prompt": inpaint_prompt,
            "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, disfigured, gross proportions",
            "steps": 30,
            "sampler": "ddim",
            "seed": -1,
            "count": 1
        }
        return inpaint(item)
    else:
        return None
