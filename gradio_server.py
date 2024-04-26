import sys
import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from ai_translator.utils import ArgumentParser, LOG
from ai_translator.translator import PDFTranslator2, TranslationConfig


def translation(input_file, source_language, target_language):
    LOG.debug(f"[翻译任务]\n源文件: {input_file.name}\n源语言: {source_language}\n目标语言: {target_language}")

    output_file_path = Translator.translate_pdf(
        input_file.name, source_language=source_language, target_language=target_language)

    return output_file_path


def launch_gradio():

    iface = gr.Interface(
        fn=translation,
        title="OpenAI-Translator v2.0(PDF 电子书翻译工具)",
        inputs=[
            gr.File(label="上传PDF文件"),
            gr.Textbox(label="源语言（默认：英文）", placeholder="English", value="English"),
            gr.Textbox(label="目标语言（默认：中文）", placeholder="Chinese", value="Chinese")
        ],
        outputs=[
            gr.File(label="下载翻译文件")
        ],
        allow_flagging="never"
    )

    iface.launch(share=True, server_name="0.0.0.0")


def initialize_translator():
    # 解析命令行
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    # 初始化配置单例
    config = TranslationConfig()
    config.initialize(args)    
    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    global Translator

    if args.model_type == "GLMModel":
        model = ChatZhipuAI(model_name='glm-4', temperature=0, verbose=True)
    else:
        model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, verbose=True)

    Translator = PDFTranslator2(model)


if __name__ == "__main__":
    # 初始化 translator
    initialize_translator()
    # 启动 Gradio 服务
    launch_gradio()
