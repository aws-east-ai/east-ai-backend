import boto3

translate_client = boto3.client("translate")


def translate(phrase: str):
    if not phrase:
        return None
    return translate_client.translate_text(
        Text=phrase, SourceLanguageCode="auto", TargetLanguageCode="en"
    )["TranslatedText"]


def extract_keywords(phrase: str):
    if not phrase:
        return None
    return translate_client.translate_text(
        Text=phrase, SourceLanguageCode="auto", TargetLanguageCode="en"
    )["Keywords"]
