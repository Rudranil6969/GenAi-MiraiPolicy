import os
from rag_pipeline import ask_question

tests = [
    {
        "name": "Test 1 - Attendance grading",
        "question": "I have 72% attendance. How many attendance marks will I get?",
        "expected": "4 marks"
    },
    {
        "name": "Test 2 - Medical leave",
        "question": "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
        "expected": "Yashaswini Ma'am within exactly 7 days. Email not provided."
    },
    {
        "name": "Test 3 - Cybersecurity society",
        "question": "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
        "expected": "No, 40% batch support required, submit to Faculty Coordinator first."
    },
    {
        "name": "Test 4 - Unsupported monetary fine",
        "question": "How much is the fine for smoking a cigarette on campus?",
        "expected": "No monetary fine specified, leads to Disciplinary Committee action."
    }
]

def run_tests():
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not found in environment. Tests will fail if the model is not accessible.")
        
    print("Running Certification Audit (Test Suite)\n" + "="*40)
    for i, test in enumerate(tests):
        print(f"\n{test['name']}")
        print(f"Question: {test['question']}")
        print(f"Expected concept: {test['expected']}")
        try:
            result = ask_question(test["question"])
            print(f"\nGenerated Answer:\n{result['answer']}")
            print("\nSources retrieved:")
            for src in result['sources']:
                print(f"- {src['source']} (Page {src['page']})")
        except Exception as e:
            print(f"\nError: {str(e)}")
            
if __name__ == "__main__":
    run_tests()
