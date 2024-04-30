import time
import os
import gradio as gr
import jwt
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
from ai_translator.utils import ArgumentParser, LOG
from ai_translator.translator import PDFTranslator2, TranslationConfig, FileFormat


def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


# 备用：使用ChatOpenAI类调用glm的候补方式 alternative
alt_glm_model = ChatOpenAI(
    model_name='glm-4',
    temperature=0.5,  # (0.0, 1.0) 不能等于0
    verbose=True,
    openai_api_key=generate_token(os.environ['ZHIPUAI_API_KEY'], 3600),
    openai_api_base='https://open.bigmodel.cn/api/paas/v4'
)
glm_model = ChatZhipuAI(
    model_name='glm-4',
    temperature=0.5,  # (0.0, 1.0) 不能等于0
    verbose=True,
)
gpt_model = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.5,
    verbose=True,
    max_tokens=4096
)

glm_translator = PDFTranslator2(glm_model)
gpt_translator = PDFTranslator2(gpt_model)


def translation(input_file, source_language, target_language, file_format):
    LOG.debug(f"[翻译任务]\n源文件: {input_file.name}\n源语言: {source_language}\n目标语言: {target_language}")

    output_file_path = Translator.translate_pdf(
        input_file.name,
        output_file_format=file_format,
        source_language=source_language,
        target_language=target_language
    )

    return output_file_path


def launch_gradio():

    iface = gr.Interface(
        fn=translation,
        title=f"OpenAI-Translator v2.0(PDF 电子书翻译工具 - {model_name})",
        inputs=[
            gr.File(label="上传PDF文件"),
            gr.Textbox(label="源语言（默认：英文）", placeholder="English", value="English"),
            gr.Textbox(label="目标语言（默认：中文）", placeholder="Chinese", value="Chinese"),
            gr.Dropdown(
                label='输出文件格式', choices=[('markdown', FileFormat.MARKDOWN.value), ('pdf', FileFormat.PDF.value)],
                value=FileFormat.MARKDOWN.value
            )
        ],
        outputs=[
            gr.File(label="下载翻译文件")
        ],
        allow_flagging="never"
    )

    iface.launch(share=True, server_name="0.0.0.0")


def launch_gradio2():
    with gr.Blocks() as demo:
        gr.Markdown("""# OpenAI-Translator v2.0(PDF 电子书翻译工具)""")
        with gr.Row():
            with gr.Column():
                src_file = gr.File(label="上传PDF文件")
                with gr.Row():
                    src_lang = gr.Textbox(label="源语言（默认：英文）", placeholder="English", value="English")
                    tgt_lang = gr.Textbox(label="目标语言（默认：中文）", placeholder="Chinese", value="Chinese")
                    file_format = gr.Radio(
                        label='输出文件格式',
                        choices=[('markdown', FileFormat.MARKDOWN.value), ('pdf', FileFormat.PDF.value)],
                        value=FileFormat.MARKDOWN.value
                    )
                llm_model = gr.Radio(
                    label='大语言模型',
                    choices=['gpt-3.5-turbo', 'glm-4'],
                    value=model_name
                )
                with gr.Row():
                    clear_btn = gr.ClearButton(value='重置')
                    submit_btn = gr.Button("开始翻译", variant='primary')

            with gr.Column():
                tgt_file = gr.File(label="下载翻译文件")

        @gr.on(triggers=[submit_btn.click],
               inputs=[src_file, src_lang, tgt_lang, file_format, llm_model],
               outputs=tgt_file)
        def translation_callback(input_file, source_language, target_language, f_format, llm_name):
            global Translator
            if llm_name == 'glm-4':
                Translator = glm_translator
            else:
                Translator = gpt_translator
            return translation(input_file, source_language, target_language, f_format)

        @gr.on(triggers=[clear_btn.click],
               inputs=[],
               outputs=[src_file, src_lang, tgt_lang, file_format, llm_model, tgt_file])
        def clear():
            return None, 'English', 'Chinese', FileFormat.MARKDOWN.value, model_name, None

    demo.launch(server_name="0.0.0.0")


def initialize_translator():
    # 解析命令行
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    # 初始化配置单例
    config = TranslationConfig()
    config.initialize(args)    
    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    global Translator
    global model_name

    if args.model_type == "GLMModel":
        model_name = glm_model.model_name
        Translator = glm_translator
    else:
        model_name = gpt_model.model_name
        Translator = gpt_translator


if __name__ == "__main__":
    # 初始化 translator
    initialize_translator()
    # 启动 Gradio 服务1 webui2-1.png
    # launch_gradio()
    # 启动 Gradio 服务2 webui2-2.png
    launch_gradio2()
