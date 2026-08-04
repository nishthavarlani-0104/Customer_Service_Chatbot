import sys
import json
import os 
from pathlib import Path
import config
from vector_store_setup import get_vector_store, get_embeddings
from customer_support_chat import create_rag_chain
import uuid
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import(faithfulness,context_precision,context_recall,answer_relevancy)

from ragas.dataset_schema import EvaluationDataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from typing import List,Dict
from tqdm import tqdm 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup OpenAI client
client = OpenAI(api_key= os.getenv("GROQ_API_KEY"),base_url=os.getenv("GROQ_URL"))

# Setup LLM for evaluation
llm = llm_factory("openai/gpt-oss-120b", client=client)

# Setup embeddings for AnswerRelevancy metric
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Step-1 load the test cases
def load_test_cases(json_file_path:str) -> List[Dict]:
    with open(json_file_path,"r",encoding="utf-8") as f:
        return json.load(f)


def create_ragas_dataset(test_cases:List[Dict],rag_chain, retriever):
    user_inputs=[]
    responses=[]
    retriever_context_list=[]
    references=[]

    for each_case in test_cases:
        user_input=each_case['user_input']
        retrieved_docs=retriever.invoke(user_input)
        contexts=[doc.page_content for doc in retrieved_docs]
        response=rag_chain.invoke({
            "input":user_input},
            config ={"configurable":{"session_id":str(uuid.uuid4())}})

        answer=response
        user_inputs.append(user_input)
        responses.append(answer)
        retriever_context_list.append(contexts)
        references.append(each_case["reference"])

    ##prepare data
    data={
        "user_input":user_inputs,
        "response":responses,
        "retrieved_contexts":retriever_context_list,
        "reference":references
    }
    #create RAGAS dataset using data and correct method
    dataset=Dataset.from_dict(data)# convert dictionary to Dataset object
    return EvaluationDataset.from_hf_dataset(dataset)# convert dataset object to EvaluationDataset object


def run_ragas_evaluation(dataset:EvaluationDataset):
    metrics=[faithfulness,context_precision,context_recall,answer_relevancy]
    results= evaluate(dataset,metrics=metrics,llm=llm,embeddings=embeddings)
    return results

def main():
    print("="*60)
    print("RAG Evaluation Lab - SolarNova Dynamics Chatbot")
    print("="*60)
    print()

    try:
        BASE_DIR=Path(__file__).parent
        json_file_path=BASE_DIR / "rag_dataset.json"
        if not json_file_path.exists():
            print(f"Error JSON file not found at {json_file_path}")
            print('please ensure if rag_dataset.json exits in the same directory')
            sys.exit(1)

        test_cases=load_test_cases(str(json_file_path))    
        vector_store=get_vector_store()
        rag_chain=create_rag_chain(vector_store)

        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":2}
        )

        dataset=create_ragas_dataset(test_cases, rag_chain, retriever)

        results=run_ragas_evaluation(dataset)
        df= results.to_pandas()
        df.to_csv("ragas_evaluation_results.csv",index=False)

        print("\n" + "=" * 60)
        print("Evaluation Summary Statistics")
        print("=" * 60) 
        print()
    # Calculate mean scores for each metric
        metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
            
        for metric in metrics:
            if metric in df.columns:
                mean_score = df[metric].mean()
                std_score = df[metric].std()
                print(f"{metric.replace('_', ' ').title():<25}: {mean_score:.3f} (±{std_score:.3f})")
            
            print()
            print(f"{'Total Questions Evaluated':<25}: {len(df)}")
            print()            
            
    except Exception as e:
        print(f"\n Error during Evaluation {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
