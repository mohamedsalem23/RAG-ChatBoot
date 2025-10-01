import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

if not GOOGLE_API_KEY or not LANGCHAIN_API_KEY:
    raise ValueError("API Keys مش موجودة في .env! تأكد من الملف.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

@st.cache_resource
def build_rag_chain(pdf_folder="information"):
    # Load PDFs
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        st.error("مفيش ملفات PDF في الفولدر!")
        return None
    
    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            for split in splits:
                split.metadata["source_file"] = os.path.basename(pdf_file)
            all_splits.extend(splits)
        except Exception as e:
            st.warning(f"خطأ في تحميل {os.path.basename(pdf_file)}: {str(e)}")
    
    if not all_splits:
        st.error("مفيش محتوى تم تحميله!")
        return None
    
    # Vector Store
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # برمبت محسن: يطلب الرد بـMarkdown جدول لو السؤال عن جدول دراسي أو شيء يحتاج تنظيم
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "أنت مساعد الموارد البشرية (HR) في كلية الذكاء الاصطناعي بجامعة كفر الشيخ. "
                "أجب على أسئلة المستخدمين المتعلقة بالسياسات، الإجراءات، أو الجداول الدراسية بناءً على السياق المقدم أدناه فقط. "
                "إذا كان السؤال يتعلق بجدول دراسي أو بيانات منظمة (مثل أوقات، مواد، أماكن)، "
                "أجب بتنسيق Markdown جدول واضح (استخدم | للخلايا، --- للخط الفاصل، واجعل العناوين واضحة). "
                "إذا لم يكن جدولاً، أجب بنص عادي. "
                "السياق قد يحتوي على جداول دراسية غير منظمة أو نصوص فوضوية، لذا قم بتنظيم الإجابة بطريقة واضحة. "
                "إذا كان السياق غير كافٍ، قول 'لا يوجد معلومات كافية' مع ذكر المصدر. "
                "لا تستخدم معلومات خارجية. أجب باللغة العربية الفصحى أو اللهجة المصرية إذا لزم الأمر، "
            ),
            HumanMessagePromptTemplate.from_template("""
            السياق:
            {context}

            السؤال:
            {question}
            """)
        ]
    )
    
    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    parser = StrOutputParser()
    
    # RAG Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | parser
    )
    
    return rag_chain
