import json


async def claude2_bot(bedrock, websocket, prompt: str, history):
    modelId = "anthropic.claude-v2"
    accept = "*/*"
    contentType = "application/json"
    body = json.dumps(
        {
            "prompt": claude_combine_history(history, prompt),
            "max_tokens_to_sample": 2048,
            "temperature": 0.5,
            "top_p": 0.9,
        }
    )

    response = bedrock.invoke_model_with_response_stream(
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
                # print(chunk_obj)
                text = chunk_obj["completion"]
                await websocket.send_text(text)

        result_end = '{"status": "done"}'
        await websocket.send_text(result_end)


async def claude3_bot(bedrock, websocket, prompt: str, history):
    accept = "*/*"
    contentType = "application/json"
    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": claude_combine_history(history, prompt),
                        },
                    ],
                }
            ],
        }
    )

    response = bedrock.invoke_model_with_response_stream(
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
                if chunk_obj["type"] == "content_block_delta":
                    text = chunk_obj["delta"]["text"]
                    await websocket.send_text(text)

        result_end = '{"status": "done"}'
        await websocket.send_text(result_end)


def claude_combine_history(history, newQ):
    if not history:
        return "Human:{prompt} \\n\\nAssistant:".format(prompt=newQ)

    prompt = ""
    for [q, a] in history:
        prompt = prompt + "Human:{q}\\n\\nAssistant:{a}\\n\\n".format(q=q, a=a)
    prompt = prompt + "Human:{prompt} \\n\\nAssistant:".format(prompt=newQ)
    # print(prompt)
    return prompt
