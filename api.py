import json
import requests

from volcengine.auth.SignerV4 import SignerV4
from volcengine.base.Request import Request
from volcengine.Credentials import Credentials

collection_name = "test"
project_name = "default"
# resource_id = "kb-3f1a5503aec6eadd"
g_knowledge_base_domain = "api-knowledgebase.ml_platform.cn-beijing.volces.com"
account_id = "2105180422"
ak = "AKLTNjAxZDE4MDc4M2Y0NGNjZmExN2Y1NzYxYTE1MTUxZWM"
sk = "TUdVNU16RXdaak01TXpnMU5EazNNemt5T0RreVpXWmpZalpsT1RJelpUYw=="

def prepare_request(method, path, params=None, data=None, doseq=0):
    if params:
        for key in params:
            if (
                    isinstance(params[key], int)
                    or isinstance(params[key], float)
                    or isinstance(params[key], bool)
            ):
                params[key] = str(params[key])
            elif isinstance(params[key], list):
                if not doseq:
                    params[key] = ",".join(params[key])
    r = Request()
    r.set_shema("http")
    r.set_method(method)
    r.set_connection_timeout(10)
    r.set_socket_timeout(10)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Host": g_knowledge_base_domain,
        "V-Account-Id": account_id,
    }
    r.set_headers(headers)
    if params:
        r.set_query(params)
    r.set_host(g_knowledge_base_domain)
    r.set_path(path)
    if data is not None:
        r.set_body(json.dumps(data))

    # 生成签名
    credentials = Credentials(ak, sk, "air", "cn-north-1")
    SignerV4.sign(r, credentials)
    return r

def search_knowledge(base_prompt, query, limit=10, dense_weight=0.5, history_messages=[], rewrite=False):
    method = "POST"
    path = "/api/knowledge/collection/search_knowledge"
    
    # 构建messages列表，包含历史记录
    messages = [{"role": "system", "content": base_prompt}]
    
    # 添加历史消息
    history_len = len(history_messages)
    if history_len > 0:
        for i, msg in enumerate(history_messages):
            # 跳过系统消息，系统消息已经添加
            if msg["role"] == "system":
                continue
            # 添加其他所有历史消息
            messages.append(msg)
    # print(messages)
    request_params = {
    "project": project_name,
    "name": collection_name,
    "query": query,
    "limit": limit,
    "pre_processing": {
        "need_instruction": True,
        "return_token_usage": True,
        "messages": messages,
        "rewrite": rewrite
    },
    "dense_weight": dense_weight,
    "post_processing": {
        "get_attachment_link": True,
        "rerank_only_chunk": False,
        "rerank_switch": False,
        "chunk_group": True,
        "chunk_diffusion_count": 0
    }
}

    info_req = prepare_request(method=method, path=path, data=request_params)
    rsp = requests.request(
        method=info_req.method,
        url="http://{}{}".format(g_knowledge_base_domain, info_req.path),
        headers=info_req.headers,
        data=info_req.body
    )
    # print("search res = {}".format(rsp.text))
    return rsp.text

def chat_completion(message, stream=True, return_token_usage=True, temperature=0.7, max_tokens=4096, model="Doubao-1-5-pro-32k", model_version="250115"):
    method = "POST"
    path = "/api/knowledge/chat/completions"
    request_params = {
        "messages": message,
        "stream": stream,
        "return_token_usage": return_token_usage,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model_version": model_version
    }

    info_req = prepare_request(method=method, path=path, data=request_params)
    rsp = requests.request(
        method=info_req.method,
        url="http://{}{}".format(g_knowledge_base_domain, info_req.path),
        headers=info_req.headers,
        data=info_req.body,
        stream=stream
    )
    rsp.encoding = "utf-8"
    
    if not stream:
        print("chat completion res = {}".format(rsp.text))
        return rsp.text
    else:
        def generate_stream():
            full_response = ""
            for line in rsp.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data:"):
                        try:
                            if line.strip() == "data: [DONE]":
                                print("Stream completed")
                                break
                                
                            json_str = line[5:]  # 去掉"data:"前缀
                            data = json.loads(json_str)
                            if "data" in data and "generated_answer" in data["data"]:
                                chunk = data["data"]["generated_answer"]
                                full_response += chunk
                                yield chunk
                        except json.JSONDecodeError:
                            pass
            
            print(f"完整回复: {full_response}")
            
        return generate_stream()

def is_vision_model(model_name):
    if model_name is None:
        return False
    return "vision" in model_name
def get_content_for_prompt(point: dict, image_num: int) -> str:
    content = point["content"]
    original_question = point.get("original_question")
    if original_question:
        # faq 召回场景，content 只包含答案，需要把原问题也拼上
        return "当询问到相似问题时，请参考对应答案进行回答：问题：“{question}”。答案：“{answer}”".format(
                question=original_question, answer=content)
    if image_num > 0 and "chunk_attachment" in point and point["chunk_attachment"][0]["link"]: 
        placeholder = f"<img>图片{image_num}</img>"
        return content + placeholder
    return content

def generate_prompt(rsp_txt, base_prompt):
    rsp = json.loads(rsp_txt)
    if rsp["code"] != 0:
        return "", []
    prompt = ""
    image_urls = []
    rsp_data = rsp["data"]
    points = rsp_data["result_list"]
    using_vlm = is_vision_model("Doubao-1-5-pro-32k")
    image_cnt = 0

    for point in points:
        # 提取图片链接
        if using_vlm and "chunk_attachment" in point:
            image_link = point["chunk_attachment"][0]["link"]
            if image_link:
                image_urls.append(image_link)
                image_cnt += 1
        # 先拼接系统字段
        doc_info = point["doc_info"]
        for system_field in ["doc_name","title","chunk_title","content","point_id"] : 
            if system_field == 'doc_name' or system_field == 'title':
                if system_field in doc_info:
                    prompt += f"{system_field}: {doc_info[system_field]}\n"
            else:
                if system_field in point:
                    if system_field == "content":
                        prompt += f"content: {get_content_for_prompt(point, image_cnt)}\n"
                    else:
                        prompt += f"{system_field}: {point[system_field]}\n"
        if "table_chunk_fields" in point:
            table_chunk_fields = point["table_chunk_fields"]
            for self_field in [] : 
                # 使用 next() 从 table_chunk_fields 中找到第一个符合条件的项目
                find_one = next((item for item in table_chunk_fields if item["field_name"] == self_field), None)
                if find_one:
                    prompt += f"{self_field}: {find_one['field_value']}\n"

        prompt += "---\n"

    return base_prompt.format(prompt), image_urls


def search_knowledge_and_chat_completion(base_prompt, query, config, history_messages=[]):
    # 1.执行search_knowledge
    rsp_txt = search_knowledge(base_prompt, query, config["limit"], config["dense_weight"], history_messages, config["rewrite"])
    # 2.生成prompt
    prompt, image_urls = generate_prompt(rsp_txt, base_prompt)
    if config["rewrite"]:
        rsp = json.loads(rsp_txt)
        query = rsp["data"]["rewrite_query"]
    
    # 3.构建消息列表，包含历史消息
    # 首先添加系统提示
    messages = [{"role": "system", "content": prompt}]
    
    history_len = len(history_messages)
    if history_len > 0:
        # 过滤并添加有效的历史消息
        for i, msg in enumerate(history_messages):
            # 跳过系统消息
            if msg["role"] == "system":
                continue
            if i == history_len - 1 and msg["role"] == "user":
                continue
            messages.append(msg)
    
    # 添加当前用户查询
    if image_urls:
        multi_modal_msg = [{"type": "text", "text": query}]
        for image_url in image_urls:
            multi_modal_msg.append({"type": "image_url", "image_url": {"url": image_url}})
        messages.append({"role": "user", "content": multi_modal_msg})
    else:
        messages.append({"role": "user", "content": query})
    
    # 4.调用chat_completion
    return chat_completion(messages, stream=True, temperature=config["temperature"], max_tokens=config["max_tokens"], model=config["model"], model_version=config["model_version"])

def direct_chat_completion(system_prompt, user_prompt, config, history_messages=[]):
    """
    Performs a direct chat completion call without knowledge search or prompt augmentation.
    Uses the provided system_prompt and user_prompt directly.
    Handles history similarly to the RAG function if provided.
    """
    # 1. Construct messages
    messages = [{"role": "system", "content": system_prompt}]

    # Add history (optional) - Use similar logic as RAG function if needed
    history_len = len(history_messages)
    if history_len > 0:
        for i, msg in enumerate(history_messages):
            if msg["role"] == "system":
                continue
            messages.append(msg)

    # 2. Add current user prompt
    # Check if user_prompt could be multimodal (unlikely for online prompt gen, but for generality)
    if isinstance(user_prompt, list): # Assume list means multimodal content
         messages.append({"role": "user", "content": user_prompt})
    else:
         messages.append({"role": "user", "content": user_prompt})

    # 3. Call chat_completion
    # Use relevant parts of the config. Assume stream=False for short prompts unless specified.
    # Extract necessary params from config to avoid passing unrelated ones like limit/dense_weight
    
    rsp_txt = chat_completion(
        message=messages,
        stream=config.get("stream", False), # Default to False for direct calls, allow override
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 256), # Sensible default for short prompts
        model=config["model"],
        model_version=config["model_version"],
        return_token_usage=config.get("return_token_usage", True)
    )
    rsp = json.loads(rsp_txt)
    if rsp["code"] != 0:
        return "", []
    return rsp["data"]["generated_answer"]
