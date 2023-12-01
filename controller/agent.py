# import os
from fastapi import APIRouter
from utils.agent import parse_task
from utils.painter import product_design, inpaint
from utils.aws import translate


class Agent:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/api/agent", self.route_agent, methods=["POST"])

    def route_agent(self, item: dict):
        assert "prompt" in item
        response = parse_task(item["prompt"])
        task_name = response["name"]

        print(response)

        if task_name == "image-generator":
            prompts = translate(response["parameters"]["prompts"])
            item = {
                "prompt": "3D product render, {}, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k".format(
                    prompts
                ),
                "negative_prompt": "EasyNegative, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), cropped, text, jpeg artifacts, signature, watermark, username, sketch, cartoon, drawing, anime, duplicate, blurry, semi-realistic, out of frame, ugly, deformed",
                "steps": 30,
                "sampler": "ddim",
                "seed": -1,
                "height": 512,
                "width": 512,
                "count": 1,
            }
            result = product_design(item)
            result["object"] = response["parameters"]["prompts"]
            return result

        elif response["name"] == "change-background":
            input_image = response["parameters"]["url"]
            sam_prompt = response["parameters"]["object"]
            inpaint_prompt = response["parameters"]["prompts"]
            item = {
                "input_image": input_image,
                "sam_prompt": sam_prompt,
                "prompt": inpaint_prompt,
                "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, disfigured, gross proportions",
                "steps": 30,
                "sampler": "ddim",
                "seed": -1,
                "count": 1,
            }
            return inpaint(item)
        else:
            return {"success": False, "message": "Can not find task."}


agent_router = Agent().router
