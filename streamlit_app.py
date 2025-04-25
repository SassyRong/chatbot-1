import streamlit as st
  

default_config = {
    "limit": 30,
    "temperature": 0.7,
    "max_tokens": 4096,
    "model": "Doubao-1-5-pro-32k",
    "model_version": "250115",
    "dense_weight": 0.5,
    "rewrite": False
}

# base_prompt = """# 任务
# 你是一位在线客服，你的首要任务是通过巧妙的话术回复用户的问题，你需要根据「参考资料」来回答接下来的「用户问题」，这些信息在 <context></context> XML tags 之内，你需要根据参考资料给出准确，简洁的回答。

# 你的回答要满足以下要求：
#     1. 回答内容必须在参考资料范围内，尽可能简洁地回答问题，不能做任何参考资料以外的扩展解释。
#     2. 回答中需要根据客户问题和参考资料保持与客户的友好沟通。
#     3. 如果参考资料不能帮助你回答用户问题，告知客户无法回答该问题，并引导客户提供更加详细的信息。
#     4. 为了保密需要，委婉地拒绝回答有关参考资料的文档名称或文档作者等问题。

# # 任务执行
# 现在请你根据提供的参考资料，遵循限制来回答用户的问题，你的回答需要准确和完整。

# # 参考资料
# <context>
#   {}
# </context>


# # 引用要求
# 1. 当可以回答时，在句子末尾适当引用相关参考资料，每个参考资料引用格式必须使用<reference>标签对，例如: <reference data-ref="{{point-id}}" data-img-ref="..."></reference>
# 2. 当告知客户无法回答时，不允许引用任何参考资料
# 3. 'data-ref' 字段表示对应参考资料的 point_id
# 4. 'data-img-ref' 字段表示句子是否与对应的图片相关，"true"表示相关，"false"表示不相关
# """
base_prompt = """# 任务
你是一位工作助手，你的首要任务是详细地列出解决用户问题的步骤。

你需要根据「参考资料」来回答接下来的「用户问题」，这些信息在 <context></context> XML tags 之内。

你的回答要满足以下要求:
    1. 回答内容必须在参考资料范围内，不能做任何参考资料以外的扩展解释。
    2. 将问题拆分成多个子任务，逐步解决子任务，从而最终解决用户问题。表述时可以使用如”第一步...第二步...“或者”首先...其次...“等合理的引导词。
    3. 如果用户的问题比较模糊，无法分辨用户具体询问内容，需明确告知客户无法回答，并礼貌引导客户提供更加详细的信息。
    4. 为了保密需要，若用户询问有关参考资料的文档名称或文档作者等问题，要委婉且清晰地拒绝回答。

# 任务执行
现在请你根据提供的参考资料，遵循限制来回答用户的问题，你的回答需要准确和完整。

<context>
  {}
</context>

# 引用要求
1. 当可以回答时，在句子末尾适当引用相关参考资料，每个参考资料引用格式必须使用<reference>标签对，例如: <reference data-ref="{{point-id}}" data-img-ref="..."></reference>
2. 当告知客户无法回答时，不允许引用任何参考资料
3. 'data-ref' 字段表示对应参考资料的 point_id
4. 'data-img-ref' 字段表示句子是否与对应的图片相关，"true"表示相关，"false"表示不相关
"""

from api import search_knowledge_and_chat_completion
import json

# 设置页面标题和布局
st.set_page_config(page_title="智能客服系统", layout="wide")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "config" not in st.session_state:
    st.session_state.config = default_config
if "base_prompt" not in st.session_state:
    st.session_state.base_prompt = base_prompt
if "generating" not in st.session_state:
    st.session_state.generating = False


# 定义可用的模型和版本
models_and_versions = {
    "Doubao-1-5-pro-32k": ["250115"],
    "Doubao-1-5-pro-256k": ["250115"],
    "Doubao-1-5-lite-32k": ["250115"],
    # "DeepSeek-r1-64k": ["250120"],
    # "DeepSeek-v3-128k": ["250324", "241226"],
    "Doubao-pro-32k": ["241215", "240828", "character-241215", "character-240828", "character-240528"],
    "Doubao-pro-128k": ["240628"],
    "Doubao-pro-256k": ["240828"],
    "Doubao-lite-32k": ["240828"],
    "Doubao-lite-128k": ["240828", "240515"],
    "Doubao-1-5-vision-pro-32k": ["250115"],
    "Doubao-vision-pro-32k": ["241028"]
}

# 为不同模型设置最大输出token限制
model_max_tokens = {
    "Doubao-1-5-pro-32k": 12288 ,
    "Doubao-1-5-pro-256k": 12288,
    "Doubao-1-5-lite-32k": 12288,
    # "DeepSeek-r1-64k": 8192,
    # "DeepSeek-v3-128k": 8192,
    "Doubao-pro-32k": 4096,
    "Doubao-pro-128k": 4096,
    "Doubao-pro-256k": 4096,
    "Doubao-lite-32k": 4096,
    "Doubao-lite-128k": 4096,
    "Doubao-1-5-vision-pro-32k": 12288,
    "Doubao-vision-pro-32k": 4096
}

# 设置界面切换函数
def toggle_settings():
    st.session_state.show_settings = not st.session_state.show_settings

# 清空对话历史函数
def clear_chat_history():
    st.session_state.messages = []
    st.session_state.generating = False

# 重置提示词函数
def reset_prompt():
    st.session_state.base_prompt = base_prompt

# 保存设置函数
def save_settings(model, version, temp, limit, max_tokens, dense_weight, rewrite, prompt):
    st.session_state.config = {
        "model": model,
        "model_version": version,
        "temperature": temp,
        "limit": limit,
        "max_tokens": max_tokens,
        "dense_weight": dense_weight,
        "rewrite": rewrite
    }
    st.session_state.base_prompt = prompt
    st.session_state.show_settings = False

# 主界面
st.title("大模型实战系统")

# 侧边栏 - 设置和清空对话按钮
with st.sidebar:
    st.button("⚙️ 设置", on_click=toggle_settings, use_container_width=True, disabled=st.session_state.generating)
    st.button("🗑️ 清空对话", on_click=clear_chat_history, use_container_width=True, disabled=st.session_state.generating)

# 如果显示设置界面
if st.session_state.show_settings:
    st.header("系统设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 模型选择
        st.subheader("模型与参数")
        # 模型选择
        selected_model = st.selectbox(
            "选择模型", 
            options=list(models_and_versions.keys()),
            index=list(models_and_versions.keys()).index(st.session_state.config.get("model", default_config["model"])) # 使用 .get 提供默认值
        )
        
        # 根据选择的模型更新可用的版本
        available_versions = models_and_versions[selected_model]
        current_version = st.session_state.config.get("model_version", default_config["model_version"])
        version_index = 0
        if current_version in available_versions:
             version_index = available_versions.index(current_version)

        selected_version = st.selectbox(
            "选择模型版本",
            options=available_versions,
            index=version_index
        )
        
        # 温度设置
        temp = st.slider("温度 (Temperature)", min_value=0.0, max_value=1.0, value=st.session_state.config.get("temperature", default_config["temperature"]), step=0.01,
                         help="控制生成文本的随机性。值越高，回答越随机；值越低，回答越确定。")
        
        # 检索限制设置
        limit = st.slider("检索数量限制 (Limit)", min_value=1, max_value=200, value=st.session_state.config.get("limit", default_config["limit"]), step=1,
                          help="从知识库检索的最大相关文档数量。")
        
        # 获取当前选择模型的最大token限制
        current_model_max_tokens = model_max_tokens.get(selected_model, 8000)
        
        # 最大令牌数设置 - 根据选择的模型动态设置最大值
        max_tokens = st.slider(f"最大令牌数 (Max Tokens, 上限{current_model_max_tokens})", 
                             min_value=1, 
                             max_value=current_model_max_tokens, 
                             value=min(st.session_state.config.get("max_tokens", default_config["max_tokens"]), current_model_max_tokens), 
                             step=1,
                             help="限制模型单次生成内容的最大长度（令牌数）。根据火山引擎文档，doubao-1.5系列模型最大支持12k输出长度，deepseek系列支持16k输出长度。")
        
        # 密集权重设置
        dense_weight = st.slider("密集权重 (Dense Weight)", min_value=0.2, max_value=1.0, value=st.session_state.config.get("dense_weight", default_config["dense_weight"]), step=0.01,
                                      help="1 表示纯稠密检索 ，0 表示纯字面检索，范围 [0.2, 1]，只有在请求的知识库使用的是混合检索时有效，即索引算法为 hnsw_hybrid")
        
        # 问题改写设置
        rewrite = st.checkbox("问题改写", value=st.session_state.config.get("rewrite", default_config["rewrite"]), help="启用后，将基于历史对话对本轮问题进行改写，使其具备更完整的语义信息，检索更准确。注意：改写问题会增加检索时长和额外的 Tokens 消耗。")
    
    with col2:
        # 提示词编辑
        st.subheader("系统提示词 (System Prompt)")
        prompt_text = st.text_area("编辑系统提示词", st.session_state.base_prompt, height=500)
        
        # 恢复默认提示词按钮
        if st.button("恢复默认提示词", on_click=reset_prompt):
            pass
    
    # 保存设置按钮
    st.button(
        "保存设置", 
        on_click=save_settings, 
        args=(selected_model, selected_version, temp, limit, max_tokens, dense_weight, rewrite, prompt_text),
        type="primary"
    )

# 聊天界面
else:
    # 显示聊天历史
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        st.info("👋 欢迎使用大模型实战系统，你可以在设置中修改系统提示词，然后输入您的问题...")
    
    # 聊天输入
    if st.session_state.generating:
        # 从聊天记录中获取最新的用户消息作为当前查询
        # 确保消息列表不为空且最后一条是用户消息
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            current_query = st.session_state.messages[-1]["content"]

            # --- 执行 API 调用和流式输出 ---
            with st.chat_message("assistant"):
                response_placeholder = st.empty() # 创建一个空占位符用于流式显示
                full_response = "" # 初始化完整响应字符串

                with st.spinner("生成中..."): # 显示加载指示器
                    try:
                        # 获取除最后一条用户消息之外的聊天记录，作为上下文传递给 API
                        history_for_api = st.session_state.messages[:-1]

                        # 调用你的 API 函数
                        response_generator = search_knowledge_and_chat_completion(
                            st.session_state.base_prompt,
                            current_query,
                            st.session_state.config,
                            history_for_api # 传递处理过的历史记录
                        )

                        # 处理流式响应
                        if hasattr(response_generator, '__iter__'):
                            for chunk in response_generator:
                                full_response += chunk # 累加块到完整响应
                                response_placeholder.markdown(full_response + "▌") # 在占位符中更新显示，加个光标模拟打字效果
                            response_placeholder.markdown(full_response) # 显示最终完整响应
                        else:
                            # 处理非流式响应或错误情况
                            full_response = response_generator if response_generator else "抱歉，我无法回答这个问题。"
                            response_placeholder.markdown(full_response)

                    except Exception as e:
                        # 处理 API 调用或流式处理中的异常
                        st.error(f"生成回答时出错: {str(e)}")
                        full_response = "抱歉，处理您的问题时遇到错误。"
                        response_placeholder.markdown(full_response)

            # --- 更新状态和历史记录 ---
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # --- 重置状态并刷新 ---
            st.session_state.generating = False
            st.rerun()

        else:
            # 如果 generating 为 True 但找不到用户消息，重置状态
            st.warning("无法找到待处理的用户消息，重置状态。")
            st.session_state.generating = False
            st.rerun()

    # --- 聊天输入框 ---
    if query := st.chat_input("请输入您的问题...", disabled=st.session_state.generating, key="chat_input"):
        with st.chat_message("user"):
            st.write(query)

        st.session_state.messages.append({"role": "user", "content": query})

        st.session_state.generating = True

        st.rerun()
