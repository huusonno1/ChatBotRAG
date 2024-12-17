from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


class VectorDBManager:
    def __init__(self):
        # Cấu trúc thư mục
        self.base_dir = os.getcwd()  # Thư mục hiện tại
        self.data_dir = os.path.join(self.base_dir, "data")  # Thư mục chứa dữ liệu PDF
        self.vector_dir = os.path.join(self.base_dir, "vectorstores/db_faiss")  # Thư mục lưu vector DB

        # Tạo các thư mục nếu chưa tồn tại
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.vector_dir, exist_ok=True)

        # Khởi tạo embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    def create_db_from_text(self, raw_text: str):
        """Tạo vector DB từ văn bản"""
        try:
            # Chia nhỏ văn bản
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=512,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(raw_text)

            # Tạo và lưu vector DB
            db = FAISS.from_texts(texts=chunks, embedding=self.embedding_model)
            db.save_local(self.vector_dir)
            print(f"Vector DB đã được lưu tại: {self.vector_dir}")
            return db

        except Exception as e:
            print(f"Lỗi khi tạo vector DB: {str(e)}")
            return None

    def create_db_from_pdf(self):
        """Tạo vector DB từ các file PDF trong thư mục data"""
        try:
            # Load tất cả file PDF
            loader = DirectoryLoader(self.data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()

            # Chia nhỏ văn bản
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            chunks = text_splitter.split_documents(documents)

            # Tạo và lưu vector DB
            db = FAISS.from_documents(chunks, self.embedding_model)
            db.save_local(self.vector_dir)
            print(f"Vector DB đã được lưu tại: {self.vector_dir}")
            return db

        except Exception as e:
            print(f"Lỗi khi tạo vector DB từ PDF: {str(e)}")
            return None

    def load_vector_db(self):
        """Load vector DB đã tồn tại"""
        try:
            return FAISS.load_local(self.vector_dir, self.embedding_model)
        except Exception as e:
            print(f"Lỗi khi load vector DB: {str(e)}")
            return None


def main():
    # Text mẫu để test
    sample_text = """Nhằm đáp ứng nhu cầu và thị hiếu của khách hàng về việc sở hữu số tài khoản đẹp...
    [phần text còn lại của bạn]
    """

    # Khởi tạo VectorDBManager
    db_manager = VectorDBManager()

    # Tạo vector DB từ text
    # db = db_manager.create_db_from_text(sample_text)

    # Hoặc tạo từ PDF
    db = db_manager.create_db_from_pdf()

    # Test load lại vector DB
    # loaded_db = db_manager.load_vector_db()


if __name__ == "__main__":
    main()