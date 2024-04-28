from typing import Optional
from ai_translator.book import ContentType
from ai_translator.model import Model
from ai_translator.translator.pdf_parser import PDFParser
from ai_translator.translator.writer import Writer, FileFormat
from ai_translator.translator.translation_chain import TranslationChain, TableTranslationChain
from ai_translator.utils import LOG


class PDFTranslator:
    def __init__(self, model: Model):
        self.model = model
        self.pdf_parser = PDFParser()
        self.writer = Writer()
        self.book = None

    def translate_pdf(self, pdf_file_path: str, file_format: str = FileFormat.PDF.value, target_language: str = '中文', output_file_path: str = None, pages: Optional[int] = None):
        self.book = self.pdf_parser.parse_pdf(pdf_file_path, pages)

        for page_idx, page in enumerate(self.book.pages):
            for content_idx, content in enumerate(page.contents):
                prompt = self.model.translate_prompt(content, target_language)
                LOG.debug(prompt)
                translation, status = self.model.make_request(prompt)
                LOG.info(translation)
                
                # Update the content in self.book.pages directly
                self.book.pages[page_idx].contents[content_idx].set_translation(translation, status)

        self.writer.save_translated_book(self.book, output_file_path, file_format)


class PDFTranslator2:
    def __init__(self, model):
        self.translate_chain = TranslationChain(llm=model)
        self.table_translate_chain = TableTranslationChain(llm=model)
        self.pdf_parser = PDFParser()
        self.writer = Writer()
        self.book = None

    def translate_pdf(self,
                      input_file: str,
                      output_file_format: str = FileFormat.MARKDOWN.value,
                      source_language: str = "English",
                      target_language: str = 'Chinese',
                      pages: Optional[int] = None):

        self.book = self.pdf_parser.parse_pdf(input_file, pages)

        for page_idx, page in enumerate(self.book.pages):
            for content_idx, content in enumerate(page.contents):
                # Translate content.original
                if content.content_type == ContentType.TABLE:
                    translation, status = self.table_translate_chain.run(str(content), source_language, target_language)
                else:
                    translation, status = self.translate_chain.run(str(content), source_language, target_language)
                # Update the content in self.book.pages directly
                self.book.pages[page_idx].contents[content_idx].set_translation(translation['text'], status)

        return self.writer.save_translated_book(self.book, file_format=output_file_format)
