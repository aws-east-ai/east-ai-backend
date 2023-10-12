from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse
from datetime import datetime
from io import BytesIO
from PIL import Image
from utils.image import sd_resize_image
from utils.common import get_mime_type
import uuid
import os
import boto3


class Tools:
    def __init__(self):
        self.s3_bucket = os.environ.get("WORKSHOP_IMAGE_BUCKET", "east-ai-workshop")
        self.s3 = boto3.resource("s3")
        self.router = APIRouter()
        self.router.add_api_route("/api/upload", self.upload, methods=["POST"])

        self.router.add_api_route(
            "/api/s3-image/{s3_url:path}", self.render_image_from_s3, methods=["GET"]
        )

    async def upload(self, file: UploadFile):
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

        self.s3.Bucket(self.s3_bucket).put_object(
            Key=f"images/{key_original}", Body=img_byte_arr
        )

        return {"success": True, "data": f"s3://{self.s3_bucket}/images/{key_original}"}

    async def render_image_from_s3(self, s3_url: str):
        split_tup = os.path.splitext(s3_url)
        media_type = get_mime_type(split_tup[1])
        # 只允许特定的扩展名
        if not media_type:
            return None
        # 解析 bucket 和 key
        s3_urls = s3_url.replace("s3://", "").replace("s3:/", "").split("/")
        # print("vvv|||", s3_url, s3_urls)
        n_bucket = s3_urls[0]
        key = s3_url.replace("s3://" + self.s3_bucket + "/", "").replace(
            "s3:/" + self.s3_bucket + "/", ""
        )

        xobject = self.s3.Object(n_bucket, key).get()
        img_bytes = xobject["Body"].read()
        image_stream = BytesIO(img_bytes)
        return StreamingResponse(content=image_stream, media_type=media_type)


tools_router = Tools().router
