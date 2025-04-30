import streamlit as st
import streamlit.components.v1 as components 
from api import search_knowledge_and_chat_completion, direct_chat_completion
import json
import clipboard
from st_copy_to_clipboard import st_copy_to_clipboard

# --- Default Config and Base Prompt (Keep yours) ---
default_config = {
    "limit": 30,
    "temperature": 0.7,
    "max_tokens": 4096,
    "model": "Doubao-1-5-pro-32k",
    "model_version": "250115",
    "dense_weight": 0.5,
    "rewrite": False
}

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

online_base_prompt = """# 任务
你是一个专门生成搜索查询语句的助手。
根据用户以及检索助手提供的检索的知识库信息，生成一个精确的中文搜索查询 Prompt。
这个 Prompt 的目标是联网搜索指定人物的详细信息，包括其工作单位、研究方向、重要事件、荣誉奖项以及紧密合作者等。

# 输出信息参考
{姓名}依托于{单位名称}申请{项目类别}，项目名称为{项目名称}，联网获得这个人的信息，包括工作单位信息、研究方向、重要事件、获得荣誉奖项、与他密切合作的人员及单位等。
（这里括号内信息通过用户提供的信息生成，若信息不详则不生成）

# 输出要求
1.  请根据用户提供的信息，生成一个精确的搜索 
2.  回答中不要出现未知，某，不详等不明确字样以及{项目名称}等结构化输出，确保句子通顺可读
3.  不要添加任何额外的解释、注释或寒暄，不要出现太多项目实际信息。
4.  输出内容必须是纯文本格式的 Prompt 字符串。
5.  回答长度严格限制在 256 token 以内。
6.  多个项目或者多个负责人，请用“、”连接。
"""

VALID_CREDENTIALS = {
    "april@zju.edu.cn": "april666",
    "rag@zju.edu.cn": "rag666"
}

# --- Model Definitions (Keep yours) ---
models_and_versions = {
    "Doubao-1-5-pro-32k": ["250115"],
    "Doubao-1-5-pro-256k": ["250115"],
    "Doubao-1-5-lite-32k": ["250115"],
    "Doubao-pro-32k": ["241215", "240828", "character-241215", "character-240828", "character-240528"],
    "Doubao-pro-128k": ["240628"],
    "Doubao-pro-256k": ["240828"],
    "Doubao-lite-32k": ["240828"],
    "Doubao-lite-128k": ["240828", "240515"],
    "Doubao-1-5-vision-pro-32k": ["250115"],
    "Doubao-vision-pro-32k": ["241028"]
}

model_max_tokens = {
    "Doubao-1-5-pro-32k": 12288 ,
    "Doubao-1-5-pro-256k": 12288,
    "Doubao-1-5-lite-32k": 12288,
    "Doubao-pro-32k": 4096,
    "Doubao-pro-128k": 4096,
    "Doubao-pro-256k": 4096,
    "Doubao-lite-32k": 4096,
    "Doubao-lite-128k": 4096,
    "Doubao-1-5-vision-pro-32k": 12288,
    "Doubao-vision-pro-32k": 4096
}

# --- Page Config ---
st.set_page_config(page_title="大语言模型实战系统", layout="wide") # Changed title slightly


if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'login_error' not in st.session_state:
    st.session_state.login_error = ""
        
def initialize_app_state():
    # --- Session State Initialization ---
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
    # NEW: Initialize view mode state
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "chat" # Start in chat mode
    if "recommended_prompt" not in st.session_state:
        st.session_state.recommended_prompt = ""
    if "online_generating" not in st.session_state: # State flag for online prompt generation
        st.session_state.online_generating = False

# --- Helper Functions (Keep yours) ---
def toggle_settings():
    # Toggle settings only if in chat mode and not already showing settings
    if st.session_state.view_mode == "chat":
        st.session_state.show_settings = not st.session_state.show_settings
    # Ensure we are not in web search mode when trying to show settings
    # If somehow toggle is called from web_search, force back to chat
    # elif st.session_state.view_mode == "web_search":
    #     st.session_state.view_mode = "chat"
    #     st.session_state.show_settings = True # Or False depending on desired behavior

def clear_chat_history():
    st.session_state.messages = []
    st.session_state.generating = False
    # Make sure settings are closed after clearing
    st.session_state.show_settings = False
    # Ensure we stay/return to chat view
    st.session_state.view_mode = "chat"
    st.session_state.online_generating = False
    # No rerun needed here if called via on_click, Streamlit handles it.

def reset_prompt():
    st.session_state.base_prompt = base_prompt
    # No rerun needed here, just updates state for next save.

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
    st.session_state.show_settings = False # Hide settings after saving
    # No rerun needed here if called via on_click


def logout():
    """Clears login state and other relevant session data."""
    st.session_state.logged_in = False
    st.session_state.login_error = ""
    # Optionally clear other states upon logout
    keys_to_clear = [
        "messages", "show_settings", "config", "base_prompt", "generating",
        "view_mode", "recommended_prompt", "prompt_gen_input_name",
        "prompt_gen_input_org", "prompt_gen_input_cat", "prompt_gen_input_proj",
        "generating_rec_prompt"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key] # Remove the key entirely

def login_screen():
    st.title("登录 - 大语言模型实战系统")
    st.markdown("---")

    with st.form("login_form"):
        username_input = st.text_input("邮箱 (Email)", key="login_username")
        password = st.text_input("密码 (Password)", type="password", key="login_password")
        submitted = st.form_submit_button("登录")

        if submitted:
            # --- Modification: Strip whitespace from username ---
            username = username_input.strip()
            # ----------------------------------------------------

            if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.session_state.login_error = "" # Clear error on success
                initialize_app_state() # Initialize app state AFTER successful login
                st.rerun() # Rerun to show the main app
            else:
                st.session_state.logged_in = False
                st.session_state.login_error = "用户名或密码错误，请重试。"
                # Error will be displayed outside the form after rerun

    # Display login error message if it exists
    if st.session_state.login_error:
        st.error(st.session_state.login_error)
  
def main_app():
    initialize_app_state()
# --- Sidebar ---
    with st.sidebar:
        st.title("导航与操作")
        st.button("退出登录", on_click=logout, use_container_width=True, type="secondary")
        st.divider()

        # Button to switch to Web Search View
        if st.button("🌐 联网搜索", key="goto_web_search", use_container_width=True, disabled=st.session_state.generating or st.session_state.online_generating):
            st.session_state.view_mode = "web_search"
            st.session_state.show_settings = False # Close settings if open
            st.rerun() # Force rerun to show the web view

        # Conditional buttons for Chat mode
        if st.session_state.view_mode == "chat":
            st.divider()
            # Disable settings button if generating, otherwise use the toggle function
            st.button("⚙️ 设置", on_click=toggle_settings, use_container_width=True, disabled=st.session_state.generating or st.session_state.online_generating)
            # Disable clear button if generating
            st.button("🗑️ 清空对话", on_click=clear_chat_history, use_container_width=True, disabled=st.session_state.generating or st.session_state.online_generating)

    # --- Main Content Area ---
    st.title("大模型实战系统") # Main title visible in both views

    # --- View Routing ---
    if st.session_state.view_mode == "chat":
        # --- CHAT VIEW ---

        # --- Settings Panel (Only shown if show_settings is True) ---
        if st.session_state.show_settings:
            st.header("系统设置")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("模型与参数")
                selected_model = st.selectbox(
                    "选择模型",
                    options=list(models_and_versions.keys()),
                    index=list(models_and_versions.keys()).index(st.session_state.config.get("model", default_config["model"]))
                )
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
                temp = st.slider("温度 (Temperature)", 0.0, 1.0, st.session_state.config.get("temperature", default_config["temperature"]), 0.01, help="控制生成文本的随机性。值越高，回答越随机；值越低，回答越确定。")
                limit = st.slider("检索数量限制 (Limit)", 1, 200, st.session_state.config.get("limit", default_config["limit"]), 1, help="从知识库检索的最大相关文档数量。")
                current_model_max_tokens = model_max_tokens.get(selected_model, 8000)
                max_tokens_value = min(st.session_state.config.get("max_tokens", default_config["max_tokens"]), current_model_max_tokens)
                max_tokens = st.slider(f"最大令牌数 (Max Tokens, 上限{current_model_max_tokens})", 1, current_model_max_tokens, max_tokens_value, 1, help="限制模型单次生成内容的最大长度（令牌数）。")
                dense_weight = st.slider("密集权重 (Dense Weight)", 0.2, 1.0, st.session_state.config.get("dense_weight", default_config["dense_weight"]), 0.01, help="1 表示纯稠密检索 ，0 表示纯字面检索，范围 [0.2, 1]，只有在请求的知识库使用的是混合检索时有效，即索引算法为 hnsw_hybrid")
                rewrite = st.checkbox("问题改写", value=st.session_state.config.get("rewrite", default_config["rewrite"]), help="启用后，将基于历史对话对本轮问题进行改写，使其具备更完整的语义信息，检索更准确。注意：改写问题会增加检索时长和额外的 Tokens 消耗。")

            with col2:
                st.subheader("系统提示词 (System Prompt)")
                prompt_text = st.text_area("编辑系统提示词", st.session_state.base_prompt, height=500)
                if st.button("恢复默认提示词", on_click=reset_prompt):
                    # Re-render the text area with the reset prompt immediately
                    st.rerun() # Rerun might be needed here if you want instant update of text_area

            st.button(
                "保存设置",
                on_click=save_settings,
                args=(selected_model, selected_version, temp, limit, max_tokens, dense_weight, rewrite, prompt_text),
                type="primary",
                use_container_width=True # Make button wider
            )
            # Add a button to close settings without saving
            if st.button("取消", use_container_width=True):
                st.session_state.show_settings = False
                st.rerun()


        # --- Chat Interface (Only shown if settings are NOT shown) ---
        elif not st.session_state.show_settings:
            # Display chat history
            if st.session_state.messages:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
            else:
                st.info("👋 欢迎使用大模型实战系统，你可以在设置中修改系统提示词，然后输入您的问题...")

            # Handle response generation (if generating flag is set)
            if st.session_state.generating:
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    current_query = st.session_state.messages[-1]["content"]
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_response = ""
                        with st.spinner("生成中..."):
                            try:
                                history_for_api = st.session_state.messages[:-1]
                                response_generator = search_knowledge_and_chat_completion(
                                    st.session_state.base_prompt,
                                    current_query,
                                    st.session_state.config,
                                    history_for_api
                                )
                                if hasattr(response_generator, '__iter__'):
                                    for chunk in response_generator:
                                        full_response += chunk
                                        response_placeholder.markdown(full_response + "▌")
                                    response_placeholder.markdown(full_response)
                                else:
                                    full_response = response_generator if response_generator else "抱歉，我无法回答这个问题。"
                                    response_placeholder.markdown(full_response)
                            except Exception as e:
                                st.error(f"生成回答时出错: {str(e)}")
                                full_response = "抱歉，处理您的问题时遇到错误。"
                                response_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.generating = False
                    st.rerun() # Rerun after generation is complete

                else:
                    # Handle case where generating is True but no user message found (edge case)
                    st.warning("无法找到待处理的用户消息，重置状态。")
                    st.session_state.generating = False
                    st.rerun()

            # --- Chat Input Box (Only show if NOT generating) ---
            if query := st.chat_input("请输入您的问题...", disabled=st.session_state.generating, key="chat_input"):
                with st.chat_message("user"):
                    st.write(query)
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.generating = True # Set generating flag
                st.rerun() # Rerun to trigger the generation logic block
    elif st.session_state.view_mode == "web_search":
        # --- WEB SEARCH VIEW ---
        st.header("联网搜索")
        st.subheader("推荐Prompt生成")
        st.markdown("根据**最近一次助手回答**生成用于搜索人物信息的推荐Prompt。")

        # --- Handle Online Prompt Generation ---
        if st.session_state.online_generating:
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                last_assistant_message = str(st.session_state.messages[-1]["content"]) # Ensure it's a string

                # Prepare config specifically for direct_chat_completion
                online_config = {
                    "model": st.session_state.config["model"], # Use same model
                    "model_version": st.session_state.config["model_version"], # Use same version
                    "temperature": 0.1, # Lower temp for more deterministic prompt
                    "max_tokens": 256, # Strict limit
                    "stream": False # Expect short, non-streamed response
                }

                with st.spinner("正在调用API生成推荐 Prompt..."):
                    generated_prompt = ""
                    try:
                        # *** CALL THE NEW FUNCTION ***
                        response_text = direct_chat_completion(
                            system_prompt=online_base_prompt,
                            user_prompt=last_assistant_message,
                            config=online_config,
                            history_messages=[] # No history needed for this specific task
                        )
                        # Since stream=False, response_text should be the final string
                        generated_prompt = response_text.strip()
                        st.session_state.recommended_prompt = generated_prompt

                    except Exception as e:
                        st.error(f"生成推荐Prompt时出错: {e}")
                        st.session_state.recommended_prompt = f"生成失败: {e}"
                    finally:
                        st.session_state.online_generating = False
                        # Rerun required to update UI (remove spinner, show prompt/error)
                        st.rerun()
            else:
                st.warning("无法生成推荐Prompt，因为聊天记录为空或最后一条消息不是助手回答。请先进行对话。")
                st.session_state.recommended_prompt = ""
                st.session_state.online_generating = False
                # No rerun needed here, warning is sufficient


        # --- Display Area and Buttons ---
        st.text_area(
            "生成的Prompt:",
            value=st.session_state.recommended_prompt,
            key="prompt_display_area",
            height=150,
            help="点击“生成推荐Prompt”按钮，系统将根据最近的助手回答尝试生成搜索Prompt。点击“复制Prompt”按钮复制代码。",
            disabled=st.session_state.online_generating # Disable text area while generating
        )

        gen_col, copy_col = st.columns(2)

        with gen_col:
            if st.button("生成推荐Prompt", use_container_width=True, key="generate_rec_prompt_api",
                        disabled=st.session_state.online_generating or st.session_state.generating):
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                    st.session_state.online_generating = True
                    st.rerun()
                else:
                    st.warning("无法生成推荐Prompt，因为聊天记录为空或最后一条消息不是助手回答。请先进行对话。")
                    st.session_state.recommended_prompt = ""

        with copy_col:
            # if st.button("复制Prompt", use_container_width=True, key="copy_rec_prompt",
            #               disabled=st.session_state.online_generating or not st.session_state.recommended_prompt):
            #     if st.session_state.recommended_prompt:
            #         try:
            #             clipboard.copy(st.session_state.recommended_prompt)
            #             st.toast("✅ Prompt已复制到剪贴板！")
            #         except Exception as e:
            #             st.error(f"复制失败: {e}")
            #             st.warning("请确保已安装 `clipboard` 库 (`pip install clipboard`) 并且您的环境支持剪贴板操作。")
            #     else:
            #         st.warning("没有可复制的Prompt。")
            st_copy_to_clipboard(st.session_state.recommended_prompt)

        st.divider()

        if st.button("💬 返回对话", key="return_to_chat_from_web", use_container_width=True,
                    disabled=st.session_state.online_generating or st.session_state.generating):
            st.session_state.view_mode = "chat"
            st.rerun()

        # Embed the website
        st.write("---")
        try:
            components.iframe("https://www.doubao.com/chat/", height=700, scrolling=True)
        except Exception as e:
            st.error(f"无法加载嵌入页面: {e}")
            st.warning("请确保您的网络连接正常，并且目标网站允许被嵌入。")

    else:
        # Fallback
        st.error("无效的视图模式，将重置为聊天界面。")
        st.session_state.view_mode = "chat"
        st.rerun()  
        
if not st.session_state.logged_in:
    login_screen()
else:
    main_app() 
