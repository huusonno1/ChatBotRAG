from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Dict, Any
import os


class QASystem:
    def __init__(self, model_file: str, vector_db_path: str):
        """Khởi tạo QA system"""
        self.model_file = model_file
        self.vector_db_path = vector_db_path
        self.llm = self._load_llm()
        self.embedding_model = self._load_embeddings()
        self.db = self._load_vector_db()
        self.prompt = self._create_prompt()
        self.chain = self._create_qa_chain()

    def _load_llm(self):
        """Load local LLM model"""
        return CTransformers(
            model=self.model_file,
            model_type="llama",
            max_new_tokens=1024,
            temperature=0.01
        )

    def _load_embeddings(self):
        """Load embedding model"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    def _load_vector_db(self):
        """Load vector database"""
        try:
            if not os.path.exists(self.vector_db_path):
                print(f"Không tìm thấy vector DB tại {self.vector_db_path}")
                return None

            return FAISS.load_local(
                self.vector_db_path,
                self.embedding_model,
                allow_dangerous_deserialization=True  # Thêm tham số này
            )
        except Exception as e:
            print(f"Lỗi khi load vector DB: {str(e)}")
            return None

    def _create_prompt(self):
        """Tạo prompt template"""
        template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.

{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_qa_chain(self):
        """Tạo QA chain"""
        if self.db is None:
            raise ValueError("Vector DB chưa được khởi tạo")

        try:
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.db.as_retriever(
                    search_kwargs={"k": 3},
                    max_tokens_limit=1024
                ),
                return_source_documents=False,
                chain_type_kwargs={'prompt': self.prompt}
            )
        except Exception as e:
            print(f"Lỗi khi tạo QA chain: {str(e)}")
            return None
    def get_answer(self, question: str) -> Dict[str, Any]:
        """Lấy câu trả lời cho câu hỏi"""
        try:
            response = self.chain.invoke({"query": question})
            return response
        except Exception as e:
            return {"answer": f"Lỗi: {str(e)}"}


def main():
    # Cấu hình
    model_file = "models/vinallama-7b-chat_q5_0.gguf"
    vector_db_path = "vectorstores/db_faiss"

    # Khởi tạo QA system
    qa_system = QASystem(model_file, vector_db_path)

    # Test
    question = "1 + 1 = ?"
    response = qa_system.get_answer(question)

    print("\nCâu hỏi:", question)
    print("Trả lời:", response['result'] if 'result' in response else response)


if __name__ == "__main__":
    main()