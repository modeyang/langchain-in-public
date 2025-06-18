import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def custom_rag_evaluator(question: str, response: str, reference: str, llm: ChatOpenAI):
    """
    使用自定义提示词和评分规则评估 RAG 模型的回答质量。

    Args:
        question (str): 用户提出的问题。
        response (str): RAG 模型生成的回答。
        reference (str): 参考答案或标准答案。
        llm (ChatOpenAI): 用于评估的语言模型实例。

    Returns:
        dict: 包含评分和理由的 JSON 格式结果。
    """
    custom_prompt_template = """
你是一个专业的RAG对话评估助手，请根据以下标准评估回答质量：
1. **核心任务**：根据reference（参考答案或要求）判断response是否回答了user_input的问题，根据是否回答进行打分，回答有一定的角色个性或主观评价，可忽略，如包含历史对话，则需要参考历史对话
2. **评分规则**：
   - 1分：根据reference进行了回答，回答正确
   - 0.6-0.9分：基本回答但有轻微偏差/遗漏
   - 0.1-0.5分：仅部分相关或存在明显错误
   - 0分：完全未回答问题或与reference矛盾
3. **个性处理**：response中的角色语气/风格不影响评分，仅关注信息准确性
4. **输出格式**：严格使用JSON：{{"score": [0-1间的数值], "reason": "[评分依据]"}}

评估步骤：
① 对比reference与response的核心信息
② 检查是否解决user_input的关键问题
③ 忽略非实质性的表达风格
④ 给出精确分数和简明理由

示例输出：
{{"score": 0.7, "reason": "response提及了reference中的关键点A，但遗漏了重要信息B"}}

user_input: {question}
response: {response}
reference: {reference}
"""

    custom_prompt = ChatPromptTemplate.from_template(custom_prompt_template)

    # 构建评估链
    eval_chain = custom_prompt | llm | StrOutputParser()

    # 调用 LLM 进行评估
    eval_result_str = eval_chain.invoke({
        "question": question,
        "response": response,
        "reference": reference
    })

    # 清理 LLM 输出，移除可能的 Markdown 代码块标记
    cleaned_eval_result_str = eval_result_str.strip()
    if cleaned_eval_result_str.startswith("```json") and cleaned_eval_result_str.endswith("```"):
        cleaned_eval_result_str = cleaned_eval_result_str[len("```json\n"):-len("```")].strip()

    try:
        # 尝试解析 JSON 字符串
        eval_result = json.loads(cleaned_eval_result_str)
    except json.JSONDecodeError:
        # 如果解析失败，返回错误信息
        eval_result = {"score": 0, "reason": f"LLM 输出非 JSON 格式: {eval_result_str}"}

    except Exception as e:
        # 处理其他异常
        # 假设存在一个名为 logger 的日志记录器
        print(f"LLM 评估过程中发生错误: {e}, 原始 LLM 输出: {eval_result_str}") # 临时使用 print 替代 logger.exception
        eval_result = {"score": 0, "reason": f"LLM 评估过程中发生错误: {str(e)}, 原始 LLM 输出: {eval_result_str}"}

    return eval_result