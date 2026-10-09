import pandas as pd
from rag_pipeline import ask_question, get_model_name
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
import json

class EvaluationResult(BaseModel):
    score: int = Field(description="A score from 1 to 5")
    reasoning: str = Field(description="A short explanation of the score")

def evaluate_rag():
    tests = [
        {
            "question": "I have 72% attendance. How many attendance marks will I get?",
            "expected_outcome": "The system must state the student will receive 4 marks."
        },
        {
            "question": "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
            "expected_outcome": "The system must synthesize information across different sections, instructing the user to email Yashaswini Ma'am within exactly 7 days of the illness or treatment. Do not invent an email address."
        },
        {
            "question": "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
            "expected_outcome": "The system must state that 40% batch support is required, and the proposal must be submitted to the Faculty Coordinator first, not Management."
        },
        {
            "question": "How much is the fine for smoking a cigarette on campus?",
            "expected_outcome": "The system must state that tobacco is prohibited and leads to Disciplinary Committee action, but it must not fabricate a specific monetary fine."
        }
    ]

    llm = ChatGoogleGenerativeAI(model=get_model_name(), temperature=0.0)
    parser = JsonOutputParser(pydantic_object=EvaluationResult)

    eval_prompt = PromptTemplate(
        template="""You are an expert evaluator assessing a RAG system.
Evaluate the following generated answer against the given context, expected outcome, and question.
Score it from 1 to 5 based on:
- Factual accuracy against the retrieved context.
- Adherence to the expected outcome.
- Absence of hallucinations (no fabricated info).
- Penalize unsupported claims, fabricated policies, incorrect deadlines, and incorrect source references.

Question: {question}
Expected Outcome: {expected_outcome}
Retrieved Context: {context}
Generated Answer: {answer}

{format_instructions}
""",
        input_variables=["question", "expected_outcome", "context", "answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    eval_chain = eval_prompt | llm | parser

    results = []
    
    print("Starting automated evaluation...")
    for i, test in enumerate(tests):
        print(f"Evaluating Question {i+1}...")
        try:
            # Generate answer from RAG
            rag_result = ask_question(test["question"])
            answer = rag_result["answer"]
            
            # Format context string
            context_str = "\n\n".join([f"Source: {s['source']} (Page {s['page']})\n{s['content']}" for s in rag_result["sources"]])
            
            # Run evaluation
            eval_output = eval_chain.invoke({
                "question": test["question"],
                "expected_outcome": test["expected_outcome"],
                "context": context_str,
                "answer": answer
            })
            
            results.append({
                "Question": test["question"],
                "Expected Outcome": test["expected_outcome"],
                "Generated Answer": answer,
                "Retrieved Context": context_str,
                "Score": eval_output["score"],
                "Reasoning": eval_output["reasoning"]
            })
            
        except Exception as e:
            print(f"Error evaluating Question {i+1}: {e}")
            results.append({
                "Question": test["question"],
                "Expected Outcome": test["expected_outcome"],
                "Generated Answer": f"ERROR: {str(e)}",
                "Retrieved Context": "",
                "Score": 0,
                "Reasoning": f"Failed to execute pipeline: {str(e)}"
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("rag_eval_scores.csv", index=False)
    print("Evaluation complete. Results saved to rag_eval_scores.csv.")

if __name__ == "__main__":
    evaluate_rag()
