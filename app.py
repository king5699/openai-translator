import gradio as gr
import os
from ai_translator.translator.pdf_parser import PDFParser
from ai_translator.model import OpenAIModel
from ai_translator.book.content import ContentType

# os.environ['no_proxy'] = 'localhost, 127.0.0.1, 127.0.0.0/8, ::1'
pdf_parser = PDFParser()
model = OpenAIModel(model="gpt-3.5-turbo", api_key=os.environ['OPENAI_API_KEY'])
headers = {
    ContentType.TEXT.value: '========== [TEXT] ==========\n',
    ContentType.TABLE.value: '========== [TABLE] ==========\n',
}
block_break = '\n========== [END] ==========\n\n'


with gr.Blocks(css='static/app.css') as demo:
    gr.Markdown("""
    ## PDF文档在线智能翻译器
    """)
    with gr.Row():
        with gr.Column():
            source_file = gr.File(elem_id='id_source_file', label='PDF文件', file_types=['pdf'], interactive=True)
            source_textbox = gr.Textbox(elem_id='id_source_textbox', label='提示词', lines=20)

        with gr.Column():
            with gr.Row():
                with gr.Column():
                    language_dropdown = gr.Dropdown(elem_id='id_language_dropdown', choices=["中文", "日语", ],
                                                    value='中文', label="目标语言", info="")
                with gr.Column():
                    submit_btn = gr.Button("翻译", variant='primary')

            target_textbox = gr.Textbox(elem_id='id_target_textbox', label='译文', lines=20)

    @gr.on(triggers=[language_dropdown.change, source_file.upload],
           inputs=[language_dropdown, source_file],
           outputs=source_textbox)
    def upload_source_file(language, file_path):
        if not file_path:
            return ''
        book = pdf_parser.parse_pdf(file_path, pages=3)
        source_text = []
        for page in book.pages:
            for content in page.contents:
                source_text.append(headers[content.content_type.value])
                prompt = model.translate_prompt(content, language)
                source_text.append(prompt)
                source_text.append(block_break)
        return ''.join(source_text)

    @gr.on(triggers=[submit_btn.click], inputs=source_textbox, outputs=target_textbox)
    def translate(source_text):
        if not source_text.strip():
            gr.Warning('提示词不能为空')
            return ''
        try:
            target_text = []
            prompts = source_text.split(block_break)
            for prompt in prompts:
                if prompt.startswith(headers[ContentType.TEXT.value]):
                    offset = len(headers[ContentType.TEXT.value])
                elif prompt.startswith(headers[ContentType.TABLE.value]):
                    offset = len(headers[ContentType.TABLE.value])
                else:
                    continue
                translation, status = model.make_request(prompt[offset:].strip())
                if status:
                    target_text.append(translation)
            return '\n\n'.join(target_text)
        except Exception as e:
            gr.Error(f'发生异常：{e}')
            return ''


demo.launch(server_name='0.0.0.0')
