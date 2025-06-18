from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.eval_util import custom_rag_evaluator

# 假设的原始文档
original_document = """
Langchain 是一个用于开发由语言模型驱动的应用程序的框架。
它可以用于各种任务，例如问答、文本摘要和代码生成。
Langchain 提供了用于文档加载、文本分割、嵌入、检索和语言模型交互的模块。
文本分割是 RAG 中的一个关键步骤，用于将大型文档分割成更小、更易于管理的分块。
存在不同的文本分割器，包括 RecursiveCharacterTextSplitter、TokenTextSplitter 等。
文本分割的目标是创建足够小的分块，以便放入语言模型的上下文窗口中，
但同时又足够大，以保留准确回答问题所需的语义上下文。
"""
def init_chunks_with_splitter(original_document):
    """初始化分块"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.create_documents([original_document])
    return chunks



# 初始化嵌入模型
# 使用 SentenceTransformerEmbeddings 作为嵌入模型
def initialize_embeddings():
    """初始化嵌入模型"""
    # 使用 SentenceTransformers 的 all-MiniLM-L6-v2 模型
    # 这是一个轻量级的通用嵌入模型，在多个NLP任务上表现良好
    # 模型大小约为80MB，可以生成384维的向量表示
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 初始化 Chroma 向量存储并添加文档
def initialize_vectorstore(chunks, embeddings):
    """初始化 Chroma 向量存储并添加文档"""
    # 这里使用内存中的 Chroma 实例，实际应用中可以配置持久化存储
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

# 创建检索器
def create_retriever(vectorstore):
    """创建检索器"""
    return vectorstore.as_retriever()

# 创建 RAG Chain
def create_rag_chain(retriever):
    """创建 RAG Chain"""
    # 定义 Prompt 模板
    template = """
    根据以下上下文回答问题：
    {context}

    问题：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 初始化语言模型
    # 请确保已设置 OPENAI_API_KEY 环境变量
    # 从.env文件加载API密钥
    load_dotenv()
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1"
    )

    # 构建 RAG Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# 运行示例
def run_rag_example():
    """运行 RAG 示例并进行评估"""
    chunks = init_chunks_with_splitter(original_document)
    embeddings = initialize_embeddings()
    vectorstore = initialize_vectorstore(chunks, embeddings)
    retriever = create_retriever(vectorstore)
    rag_chain = create_rag_chain(retriever)

    # 示例问题和标准答案
    question = "根据文档，RAG 中文本分割的目的是什么？"
    ground_truth_answer = "RAG 中文本分割的目的是将大型文档分割成更小、更易于管理的分块，这些分块足够小，可以放入语言模型的上下文窗口中，但又足够大，以保留准确回答问题所需的语义上下文。"

    # 运行 Chain 回答问题
    response = rag_chain.invoke(question)
    print("\n问题:", question)
    print("标准答案:", ground_truth_answer)
    print("RAG 回答:", response)

    # 初始化评估器
    eval_llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1"
    )

    # 收集评估所需的数据
    # 为了简化，这里假设我们能获取到问题、RAG回答和上下文
    # 实际应用中，contexts 需要从 retriever 获取
    retrieved_docs = retriever.get_relevant_documents(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    data = {
        'question': [question],
        'answer': [response],
        'contexts': [contexts],
        'ground_truth': [ground_truth_answer]
    }

    dataset = Dataset.from_dict(data)

    print("\n开始进行 RAG 评估...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=eval_llm,
        embeddings=embeddings
    )

    print("\nRAG 评估结果:")
    print(result)

    # 运行自定义评估
    print("\n开始进行自定义 RAG 评估...")
    custom_eval_result = custom_rag_evaluator(question, response, ground_truth_answer, eval_llm)
    print("自定义 RAG 评估结果:")
    print(custom_eval_result)

# 运行示例
run_rag_example()