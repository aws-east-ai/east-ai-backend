from .aws import translate
from sagemaker import Predictor, serializers, deserializers
from .common import get_str, get_int
import os

pd_predictor = Predictor(
    endpoint_name="product-design-sd",
    serializer=serializers.JSONSerializer(),
    deserializer=deserializers.JSONDeserializer(),
)
sam_predictor = Predictor(
    endpoint_name="grounded-sam",
    serializer=serializers.JSONSerializer(),
    deserializer=deserializers.JSONDeserializer(),
)
inpaint_predictor = Predictor(
    endpoint_name="inpainting-sd",
    serializer=serializers.JSONSerializer(),
    deserializer=deserializers.JSONDeserializer(),
)

s3_bucket = os.environ.get("WORKSHOP_IMAGE_BUCKET", "east-ai-workshop")


def inpaint(item: dict):
    assert "input_image" in item
    assert "sam_prompt" in item
    output_mask_image_dir = f"s3://{s3_bucket}/mask-images/"

    sam_prompt = translate(get_str(item, "sam_prompt", None))

    mask_res = sam_predictor.predict(
        {
            "input_image": item["input_image"],
            "prompt": sam_prompt,
            "output_mask_image_dir": output_mask_image_dir,
        }
    )

    prompt = translate(get_str(item, "prompt", None))
    negative_prompt = translate(get_str(item, "negative_prompt", None))

    steps = get_int(item, "steps", 30)
    seed = get_int(item, "seed", -1)
    count = get_int(item, "count", 1)

    return inpaint_predictor.predict(
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "input_image": item["input_image"],
            "input_mask_image": mask_res["result"],
            "steps": steps,
            "sampler": item["sampler"],
            "seed": seed,
            "count": count,
        }
    )


def product_design(item: dict):
    height = get_int(item, "height", 512)
    width = get_int(item, "width", 512)

    if height % 8 != 0 or width % 8 != 0:
        return {"error": "height or width must be multiple of 64"}
    if height > 1024 or width > 1024:
        return {"error": "height or width must be less than 1024"}

    prompt = translate(get_str(item, "prompt", None))

    negative_prompt = get_str(item, "negative_prompt", None)

    if negative_prompt:
        negative_prompt = translate(negative_prompt)
    else:
        negative_prompt = "low quantity"

    steps = get_int(item, "steps", 30)
    seed = get_int(item, "seed", -1)
    count = get_int(item, "count", 1)

    item["prompt"] = prompt
    item["negative_prompt"] = negative_prompt
    item["steps"] = steps
    item["seed"] = seed
    item["height"] = height
    item["width"] = width
    item["count"] = count
    item["output_image_dir"] = f"s3://{s3_bucket}/product-images/"

    # print(item)
    return pd_predictor.predict(item)
