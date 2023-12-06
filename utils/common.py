def get_str(item: dict, key, defaultValue):
    if key in item and item[key]:
        return item[key]
    return defaultValue


def get_int(item: dict, key, defaultValue):
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


patterns = {
    "redbook": "你是一个时尚的年轻人，喜欢用emoji，请根据下面的内容写一段小红书的种草文案: ",
    "zhihu": "你是一个知识博学的学者，请根据下面的内容写一段文章，发表在知乎上: ",
    "weibo": "请根据下面的内容写一段微博的短文，140 字以内: ",
    "gongzhonghao": "你是一名思想者，请根据下面的内容写一段公众号的文章: ",
    "dianping": "你是一产品使用者，请根据下面的内容写一段点评的评论: ",
    "toutiao": "你是一个记者，请根据下面的内容写一则头条的新闻: ",
    "zhidemai": "你是一个经验丰富的导购，请根据下面的内容写一段值得买的文章: ",
    "douyin": "你是一个短片导演，请根据下面的内容写一段抖音的拍摄剧本: ",
    "kuaishou": "你是一个短片导演，请根据下面的内容写一段快手的短片剧本: ",
    "twitter": "Please write a short post on Twitter based on the following tips: ",
    "instagram": "You are a fashion influencer and love to use emojis, please write a post on Instagram based on the following tips:",
    "tiktok": "You are a stylish short film director, please write a TikTok shooting script according to the prompts: ",
    "youtube": "You are a Youtuber, please write a shooting script according to the prompts: ",
    "medium": "You are a knowledgeable writer and scientist, please follow the tips below to write a long essay, either beautifully written, or with technical depth:",
    "freestyle": "",
}


# history 格式：[[q1, a1], [q2, a2], [...]]
def claude2_chat_prompt(text, history):
    if not history:
        return "Human:{prompt} \n\nAssistant:".format(prompt=text)

    temp_prompt = ""
    for [q, a] in history:
        temp_prompt = temp_prompt + "Human:{q}\n\nAssistant:{a}\n\n".format(q=q, a=a)
    return temp_prompt + "Human:{prompt} \n\nAssistant:".format(prompt=text)
