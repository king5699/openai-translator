from langchain.chains import LLMChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from utils import LOG


class TranslationChain:
    SYSTEM_PROMPT = """You are a translation expert, proficient in various languages. \n
    Translates {source_language} to {target_language}."""

    def __init__(self, llm, verbose: bool = True):

        # 翻译任务指令始终由 System 角色承担
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT)

        # 待翻译文本由 Human 角色输入
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        self.chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=verbose)

    def run(self, text: str, source_language: str, target_language: str) -> (str, bool):
        result = ""
        try:
            result = self.chain.invoke({
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
            })
        except Exception as e:
            LOG.error(f"An error occurred during translation: {e}")
            return result, False

        return result, True


class TableTranslationChain(TranslationChain):
    SYSTEM_PROMPT = """You are a translation expert, proficient in various languages. \n
    Translates the contents of all arrays below from {source_language} to {target_language}, 
    maintain spacing (spaces, separators), and return in table form"""
