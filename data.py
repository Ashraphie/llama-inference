from datasets import load_dataset
import requests
import io
from PyPDF2 import PdfReader
from tqdm import tqdm

    
class AA_LCR:
    def __init__(self, split="test"):
        self.dataset = load_dataset("ArtificialAnalysis/AA-LCR", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "question": item["question"],
            "answer": item["answer"],
            "documents": self.__get_documents__(item["data_source_urls"].split(';'))
        }
    def __url_to_text__(self, url):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Failed to process {url}: {e}")
            return ""
    
    def __get_documents__(self, urls):
        documents = ""
        for i, url in enumerate(urls):
            text = self.__url_to_text__(url)
            if text:
                documents += f"\n\n--- Document {i+1} from {url} ---\n{text}"
        return documents.strip()
                


