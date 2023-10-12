from PIL import Image


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
