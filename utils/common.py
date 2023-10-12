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
