import json


def claude2_summuary_kb(bedrock, question, kb_contents):
    knowledges = ""
    for i in range(len(kb_contents)):
        knowledges += f"""
  <search_result>
    <content>{kb_contents[i]}</content>
    <source>{i+1}</source>
  </search_result>"""

    prompt = f"""
Human: You are a question answering agent. I will provide you with a set of search results and a user's question, your job is to answer the user's question using only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.
Here are the search results in numbered order:
<search_results>
{knowledges}
</search_results>

Here is the user's question:
<question>
{question}
</question>

If you reference information from a search result within your answer, you must include a citation to source where the information was found. Each result has a corresponding source ID that you should reference. Please output your answer in the following format:
<div meaning='answer'>
  <span meaning='answer_part'>
    <span>first answer text</span>
    <span meaning='sources'>
      <sup meaning='source'>source ID</sup>
    </span>
  </span>
  <span meaning='answer_part'>
    <span>first answer text</span>
    <span meaning='sources'>
      <upper meaning='source'>source ID</upper>
    </span>
  </span>
</div>

Note that <upper meaning='source'> may contain multiple <source> if you include information from multiple results in your answer.

Do NOT directly quote the <search_results> in your answer. Your job is to answer the <question> as concisely as possible.

Assistant:

"""

    accept = "*/*"
    contentType = "application/json"
    modelId = "anthropic.claude-v2"

    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens_to_sample": 2048,
            "temperature": 1,
            "top_p": 0.999,
            "stop_sequences": ["\n\nHuman:"],
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
                text = chunk_obj["completion"]
                yield text


def claude3_summuary_kb(bedrock, question, kb_contents):
    knowledges = ""
    for i in range(len(kb_contents)):
        knowledges += f"""
  <search_result>
    <content>{kb_contents[i]}</content>
    <source>{i+1}</source>
  </search_result>"""

    prompt = f"""
Human: You are a question answering agent. I will provide you with a set of search results and a user's question, your job is to answer the user's question using only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.
Here are the search results in numbered order:
<search_results>
{knowledges}
</search_results>

Here is the user's question:
<question>
{question}
</question>

If you reference information from a search result within your answer, you must include a citation to source where the information was found. Each result has a corresponding source ID that you should reference. Please output your answer in the following format:
<div meaning='answer'>
  <span meaning='answer_part'>
    <span>first answer text</span>
    <span meaning='sources'>
      <sup meaning='source'>source ID</sup>
    </span>
  </span>
  <span meaning='answer_part'>
    <span>first answer text</span>
    <span meaning='sources'>
      <upper meaning='source'>source ID</upper>
    </span>
  </span>
</div>

Note that <upper meaning='source'> may contain multiple <source> if you include information from multiple results in your answer.

Do NOT directly quote the <search_results> in your answer. Your job is to answer the <question> as concisely as possible.

Assistant:

"""

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
                        {"type": "text", "text": prompt},
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

    print("AAAAAAAA")
    stream = response.get("body")
    print(stream)
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            print(chunk)
            if chunk:
                chunk_obj = json.loads(chunk.get("bytes").decode())
                if chunk_obj["type"] == "content_block_delta":
                    yield chunk_obj["delta"]["text"]
    # else:
    #     yield ""
