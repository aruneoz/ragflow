import os
from random import sample
from tqdm import tqdm
import pinecone
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Pinecone
from src.data_preparation.pubmed.prompts import QUESTION_TEMPLATE


class TestsetGenerator:
    """docstring for TestsetGenerator."""

    def __init__(
        self,
        generator_llm,
        documents: list[Document],
        embedding_model: Embeddings,
        index_name: str,
        key: str = "chunk",
    ):
        self.generator_llm = generator_llm
        self.documents = documents
        self.embedding_model = embedding_model
        self.key = key
        self.prepare_splits()
        self.setup_chain()
        self.setup_vectorstore(index_name=index_name)

    def prepare_splits(self):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["."],
            chunk_size=512,
            chunk_overlap=128,
        )
        splits = text_splitter.split_documents(self.documents)
        self.splits = splits
        logger.info(f"Created {len(splits)} splits.")

    def setup_chain(self):
        prompt_template = PromptTemplate(
            template=QUESTION_TEMPLATE,
            input_variables=["context", "num_questions"],
        )
        self.chain = LLMChain(llm=self.generator_llm, prompt=prompt_template)
        logger.info("LLM and chain set up.")

    def setup_vectorstore(self, index_name):
        index = pinecone.Index(index_name=index_name)
        vectorstore = Pinecone(index, self.embedding_model, self.key)
        self.vectorstore = vectorstore

    def _parse_output(self, output):
        outputs = output.split("XXX")
        parsed_output = []
        for output in outputs:
            try:
                question, answer = output.strip().split("\n")
                question = question.replace("question:", "").strip()
                answer = answer.replace("answer:", "").strip()
            except Exception as e:
                print(f"error: {e}")
                print(output)
                print("\n===\n")

            data = {"question": question, "answer": answer}
            parsed_output.append(data)
        return parsed_output

    def _clean_synthetic_dataset(self, synthetic_dataset):
        synthetic_dataset = synthetic_dataset[
            ~(
                synthetic_dataset["question"].str.contains("the study")
                | synthetic_dataset["question"].str.contains("the research")
                | synthetic_dataset["question"].str.contains("this research")
                | synthetic_dataset["question"].str.contains("this study")
                | synthetic_dataset["question"].str.contains("studies")
                | synthetic_dataset["question"].str.contains("focus")
            )
        ]
        return synthetic_dataset

    def generate_question_and_answer(self, num_questions):
        random_split = sample(self.splits, k=1)[0]
        random_page_content = random_split.page_content
        similar_splits = self.vectorstore.similarity_search(
            random_page_content,
            k=5,
        )
        similar_page_contents = [
            similar_split.page_content for similar_split in similar_splits[1:]
        ]

        context = "\n===\n".join([random_page_content] + similar_page_contents)

        prediction = self.chain.invoke(
            {
                "context": context,
                "num_questions": num_questions,
            }
        )
        output = prediction["text"]
        parsed_outputs = self._parse_output(output)

        result = [
            {
                "question": parsed_output["question"],
                "ground_truths": [parsed_output["answer"]],
            }
            for parsed_output in parsed_outputs
        ]
        return result

    def generate(self, test_size: int, num_questions_per_context: int):
        logger.info("Creating synthetic data ...")

        data = []
        for _ in tqdm(range(test_size)):
            sample_cqa = self.generate_question_and_answer(num_questions_per_context)
            data += sample_cqa

        synthetic_dataset = pd.DataFrame(data)
        nan_questions = synthetic_dataset["question"].isnull().sum()
        nan_ratio = (nan_questions / len(data)) * 100
        logger.info(f"Synthetic dataset created. {nan_questions} ({nan_ratio:.2f})")

        old_size = len(synthetic_dataset)
        synthetic_dataset = self._clean_synthetic_dataset(synthetic_dataset)
        new_size = len(synthetic_dataset)

        if old_size == new_size:
            logger.info(f"Dataset size didn't change after post-processing: {old_size}")
        else:
            logger.info(
                f"Dataset size decreased from {old_size} to {new_size} rows after processing"
            )

        return synthetic_dataset