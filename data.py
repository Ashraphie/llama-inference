from datasets import load_dataset
import requests
import io
from PyPDF2 import PdfReader
from tqdm import tqdm
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from pathlib import Path
import json

class AA_LCR(Dataset):
    def __init__(self, split="test", from_snapshot=True):
        self.dataset = load_dataset("ArtificialAnalysis/AA-LCR", split=split)
        self.snapshot_path = Path(snapshot_download(
            repo_id="ArtificialAnalysis/AA-LCR",
            repo_type="dataset",
            local_files_only=True,
            ))
        self.local_files = (self.snapshot_path/'extracted_text'/'lcr')
        if from_snapshot:
            self.use_local_files = True
        else:
            self.use_local_files = False
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.use_local_files:
            file_paths = [self.local_files / item['document_category'] / item['document_set_id'] / file for file in item["data_source_filenames"].split(';')]
            return self._llama3_1_message_format(
                item["question"],
                self.__get_documents__(file_paths=[str(fp) for fp in file_paths]))
        return self._llama3_1_message_format(
                item["question"],
                self.__get_documents__(item["data_source_urls"].split(';'))
            )
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
    def _local_file_to_text__(self, file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                return content
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            return ""
    def __get_documents__(self, urls=None, file_paths=None):
        documents = ""
        if urls is not None and file_paths is not None:
            raise ValueError("Provide either URLs or file paths, not both.")
        if urls:
            for i, url in enumerate(urls):
                text = self.__url_to_text__(url)
                if text:
                    documents += f"\n\n--- Document {i+1} from {url} ---\n{text}"
        elif file_paths:
            for i, file_path in enumerate(file_paths):
                text = self._local_file_to_text__(file_path)
                if text:
                    documents += f"\n\n--- Document {i+1} from {file_path} ---\n{text}"
        return documents.strip()
    
    def _llama3_1_message_format(self, question, documents):
        system_prompt = (
            "You are a helpful assistant for AMD employees. Use ONLY the provided documents.\n"
            "\n"
            "Output two sections in this order:\n"
            "1) Reasoning Trace (<=8 lines total):\n"
            "   - question: one-sentence restatement.\n"
            "   - subqs: up to 3 short sub-questions.\n"
            "   - evidence: up to 3 items, each {claim, doc_id, location:'', quote<=180 chars}.\n"
            "   - synthesis: 1-2 sentences connecting the evidence to the answer (no steps or speculation).\n"
            "   - confidence: high|medium|low.\n"
            "\n"
            "2) Final Answer:\n"
            "   - Exactly one line starting with 'Final Answer: '.\n"
            "   - Single item: one string. Multiple items: newline-separated lines after the colon.\n"
            "   - If not supported by documents: 'Final Answer: None'.\n"
            "\n"
            "Rules:\n"
            "- No facts beyond the documents. Prefer precise wording for names/numbers.\n"
            "- Keep Reasoning Trace concise; do not repeat the final answer there.\n"
            "- Each non-null answer must be supported by at least one evidence item.\n"
            "- If any required part is unsupported, output None.\n"
        )

        user_prompt = f"BEGIN INPUT DOCUMENTS \n\n {documents} \n\nEND INPUT DOCUMENTS \n\nAnswer the following question using the input documents provided above. \n\nSTART QUESTION \n\n {question} END QUESTION"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

class Evaluation(Dataset):
    def __init__(self, json_file, split):
        self.dataset = load_dataset("ArtificialAnalysis/AA-LCR", split=split)
        with open(json_file, "r") as f:
            self.generation = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        generation = self.generation[idx]["answer"]
        return self.build_messages(
            question,
            item["answer"],
            generation,
        )
    def build_messages(self, question, official_answer, candidate_answer):
        system_msg = {
            "role": "system",
            "content": (
                "You are a useful model critic. Evaluate the accuracy of the model's answer based on the "
                "provided ground truth answer. You will be given a question, the model's answer, and the "
                "ground truth answer. Respond ONLY with CORRECT or INCORRECT."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                f"Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT. "
                f"For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER. "
                f"The question (for reference only): {question} "
                f"OFFICIAL ANSWER: {official_answer} "
                f"CANDIDATE ANSWER TO ASSESS: {candidate_answer} "
                f"Reply only with CORRECT or INCORRECT."
            ),
        }
        return [system_msg, user_msg]