import os
import pinecone
from langchain.chains import LLMChain
from langchain.vectorstores import Pinecone
from langchain.llms import VertexAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate


pinecone.init(
    api_key=os.environ.get("PINECONE_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)


class RAG:
    """docstring for RAG."""

    def __init__(self, index_name, llm_model_name, embedding_model, key="chunk"):
        self.index_name = index_name
        self.llm_model_name = llm_model_name
        self.embedding_model = embedding_model
        self.key = key
        self.setup_vectorstore()
        self.set_llm_chain()

    def setup_vectorstore(self):
        index = pinecone.Index(index_name=self.index_name)
        vectorstore = Pinecone(index, self.embedding_model, self.key)
        self.vectorstore = vectorstore

    def set_llm_chain(self):
        if self.llm_model_name == "text-bison":
            llm = VertexAI(
                model_name=self.llm_model_name,
                max_output_tokens=256,
                temperature=0,
                top_p=0.95,
                top_k=40,
                verbose=True,
            )
        elif self.llm_model_name == "gpt-3.5":
            llm = AzureChatOpenAI(
                deployment_name="chat-model",
                model="gpt-3.5-turbo",
            )
        elif self.llm_model_name == "gpt-4":
            llm = AzureChatOpenAI(
                deployment_name="gpt4-chat",
                model="gpt-4",
            )

        rag_prompt = PromptTemplate.from_template(RAG_TEMPLATE)
        self.chain = LLMChain(llm=llm, prompt=rag_prompt)

    def _format_source_documents(self, docs):
        source_documents = []
        for doc, score in docs:
            doc.metadata["score"] = score
            source_documents.append(doc)
        return source_documents

    def _format_context(self, docs):
        context = [doc.page_content for doc, score in docs]
        context = "\n---\n".join(context)
        return context

    def predict(self, question, top_k=4):
        relevant_documents = self.vectorstore.similarity_search_with_relevance_scores(
            query=question,
            k=top_k,
        )
        source_documents = self._format_source_documents(relevant_documents)
        context = self._format_context(relevant_documents)
        answer = self.chain.predict(question=question, context=context)
        output = {
            "question": question,
            "answer": answer,
            "source_documents": source_documents,
        }
        return output