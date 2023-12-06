import os
import json
import boto3
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from utils.aws import translate
from utils.common import get_int, get_str, claude2_chat_prompt
import random


class Bedrock:
    def __init__(self):
        self.bedrock = boto3.client("bedrock-runtime")
        self.bedrock_agent = boto3.client("bedrock-agent-runtime")
        self.knowledge_base_id = os.environ.get("KNOWLEDGE_BASE_ID")
        assert self.knowledge_base_id, "Please set env KNOWLEDGE_BASE_ID"
        self.router = APIRouter()
        self.router.add_api_route(
            "/api/bedrock-product-design", self.bedrock_product_design, methods=["POST"]
        )
        self.router.add_api_route(
            "/api/bedrock-rag", self.kb_rag_handler, methods=["POST"]
        )

    def bedrock_product_design(self, item: dict):
        # print(item)
        if "model_id" in item and item["model_id"] == "bedrock_titan":
            return self.titan_image(item)
        else:
            return self.sdxl(item)

    def titan_image(self, item: dict):
        height = get_int(item, "height", 768)
        width = get_int(item, "width", 768)

        count = get_int(item, "count", 1)
        prompt = translate(get_str(item, "prompt", None))
        negative_prompt = get_str(item, "negative_prompt", None)

        if negative_prompt:
            negative_prompt = translate(negative_prompt)
        seed = random.getrandbits(31)
        seed = seed - 2 if seed > 100 else seed
        # print(seed)
        request = json.dumps(
            {
                "textToImageParams": {"text": prompt, "negativeText": negative_prompt},
                "taskType": "TEXT_IMAGE",
                "imageGenerationConfig": {
                    "cfgScale": 8,
                    "seed": seed,
                    # "seed": 2147483646,
                    "quality": "standard",
                    "width": width,
                    "height": height,
                    "numberOfImages": count,
                },
            }
        )
        modelId = "amazon.titan-image-generator-v1"
        response = self.bedrock.invoke_model(body=request, modelId=modelId)

        response_body = json.loads(response.get("body").read())
        return response_body

    def sdxl(self, item: dict):
        height = get_int(item, "height", 768)
        width = get_int(item, "width", 768)

        # if height != 512 and width != 512:
        #     return {"error": "height or width must be 512"}
        if height % 64 != 0 or width % 64 != 0:
            return {"error": "height or width must be multiple of 64"}
        if height > 1024 or width > 1024:
            return {"error": "height or width must be less than 1024"}

        steps = get_int(item, "steps", 30)
        # item["seed"] = int(item["seed"]) or -1
        count = get_int(item, "count", 1)

        prompt = translate(get_str(item, "prompt", None))

        negative_prompt = get_str(item, "negative_prompt", None)

        if negative_prompt:
            negative_prompt = translate(negative_prompt)

        style_preset = get_str(item, "style_preset", "3d-model")
        request = json.dumps(
            {
                "text_prompts": (
                    [
                        {"text": prompt, "weight": 1.0},
                        {"text": negative_prompt, "weight": -1.0},
                    ]
                ),
                "cfg_scale": 10,
                # "seed": -1,
                "steps": steps,
                "style_preset": style_preset,
                "width": width,
                "height": height,
                "count": count,
            }
        )
        modelId = "stability.stable-diffusion-xl-v1"
        # print(request)
        response = self.bedrock.invoke_model(body=request, modelId=modelId)
        response_body = json.loads(response.get("body").read())
        # print(len(response_body["artifacts"]))
        return {"images": [response_body["artifacts"][0].get("base64")]}

    def kb_retrieve(self, text: str):
        response = self.bedrock_agent.retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={"text": text},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 5}
            },
            # nextToken="string",
        )
        # print(response)

        results = response["retrievalResults"]
        texts = []
        refs = {}
        for result in results:
            texts.append(result["content"]["text"])
            s3_url = result["location"]["s3Location"]["uri"]
            if s3_url in refs:
                refs[s3_url] = refs[s3_url] + 1
            else:
                refs[s3_url] = 1
        return texts, sorted(refs.items(), key=lambda x: x[1], reverse=True)

    async def kb_summary_content(self, text: str, history):
        # print(text)
        result = self.kb_retrieve(text)
        # print(result)
        knowledges = "\n\n".join(result[0])

        temp_prompt = ""
        if history:
            for [q, a] in history:
                a = a.replace("<br />", "\n")
                a = a[0 : a.rindex("<div class='citations'>")]
                temp_prompt = temp_prompt + "Human:{q}\n\nAssistant:{a}\n\n".format(
                    q=q, a=a
                )

        prompt = f"""Human:<history>{temp_prompt}</history>
    
    Please answer the question posed in the 'question' tag based on the information below.
    Among them, the 'history' tag is the conversation history, 
    the 'knowledge_base' tag is the knowledge try to answer current question, 
    and the knowledge in 'search_engine' comes from the internent.
    Please focus on the knowledge of 'knowledge_base', refer to the content of 'search_engine', 
    if 'history' is not empty, please focus on the questioner’s intention and whether it is related to 'history'.

    <question>{text}</question>


    <knowledge_base>{knowledges}</knowledge_base>

    <search_engine>
    </search_engine>

    Assistant:
        """

        # print(prompt)

        modelId = "anthropic.claude-v2:1"
        accept = "*/*"
        contentType = "application/json"
        body = json.dumps(
            {
                "prompt": prompt,
                "max_tokens_to_sample": 2048,
                "temperature": 1,
                "top_p": 0.999,
                "stop_sequences": ["\n\nHuman:"],
            }
        )
        response = self.bedrock.invoke_model_with_response_stream(
            body=body,
            modelId=modelId,
            accept=accept,
            contentType=contentType,
        )
        stream = response.get("body")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    chunk_obj = json.loads(chunk.get("bytes").decode())
                    text = chunk_obj["completion"]
                    yield text
        yield "<div class='citations'>Citations: "
        for s3_loc in result[1]:
            s3_full_loc = s3_loc[0][5:]
            yield "<a>" + s3_full_loc[s3_full_loc.index("/") + 1 :] + "</a>"
            yield "\n"
        yield "</div>"

    async def kb_rag_handler(self, item: dict):
        prompt = item["prompt"]
        history = item["history"] if "history" in item else []
        return StreamingResponse(self.kb_summary_content(prompt, history))


bedrock_router = Bedrock().router
