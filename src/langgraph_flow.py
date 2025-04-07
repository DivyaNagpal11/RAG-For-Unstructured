from typing import List
from typing_extensions import TypedDict
from time import sleep
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import StrOutputParser
import prompts as prompts
import warnings
from dotenv import load_dotenv
from os import environ as env
import json
from langchain_community.vectorstores import Chroma
from sentence_transformers_embedding import SentenceTransformerEmbeddings
from ollama_chat_llm import OllamaChatLLM
import glob

# Skip warning.
warnings.filterwarnings("ignore")
# load the Environment Variables.
load_dotenv()

OLLAMA_MODEL = env.get('OLLAMA_MODEL', 'mistral')
STREAM_MODE = False


class CustomDataChat():
    """
    Use Ollama models to chat with custom data.
    """
    def __init__(self) -> None:
        self.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        self.llm = OllamaChatLLM(access={
            "model": OLLAMA_MODEL
        })
        self.persist_folder = "docs/chroma_db/"
        self.store = Chroma(persist_directory=self.persist_folder,
                            embedding_function=self.embedding)

        print(self.store.__len__())


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        topic: topic
        questions: questions
        contexts: LLM generation
        report: report
    """
    topic: str
    max_queries: int
    questions: List[str]
    contexts: List[str]
    report: str


class LiteratureReviewer:
    def __init__(self,
                 max_queries: int = 5,
                 threshold: float = 0.6) -> None:
        # Max query questions
        self.max_queries = max_queries
        # Threshold for decision
        self.threshold = threshold
        # LLM dict
        self.llms = {
            # Only string output LLM
            "text": CustomDataChat().llm,
            # Grader LLM with no JSON format
            "grade": CustomDataChat().llm  # temperature=0
        }

    def plan(self, state: GraphState) -> GraphState:
        print("---PLAN---")
        topic = state["topic"]

        # Generate a search queries template
        prompt = prompts.plan_template()
        chain = prompt | self.llms["text"] | StrOutputParser()

        result = chain.invoke({"max_queries": state["max_queries"], "topic": topic})
        print(f"Result of plan: {result}")

        # Parse the JSON content
        questions_json = json.loads(result)
        questions = [q[str(i + 1)].strip() for i, q in enumerate(questions_json)]
        print(f"questions of plan : {questions}")
        return {"topic": topic, "questions": questions}

    def review_plan(self, state: GraphState) -> GraphState:
        print("---REVIEW PLAN---")
        questions = state["questions"]
        topic = state["topic"]

        # Generate a review template
        prompt = prompts.review_plan_template()
        chain = prompt | self.llms["grade"] | StrOutputParser()

        result = chain.invoke({"topic": topic, "questions": questions})
        print(f"Result of review plan: {result}")

        # Parse the JSON content
        scores = json.loads(result).get("score", {})

        print("scores of review plan", scores)
        good_questions = []
        for question, score in scores.items():
            if score == "yes":
                good_questions.append(question)

        # Update state.
        state["questions"] = good_questions
        return state

    def decide_plan(self, state: GraphState) -> str:
        """
        Decide the next action based on good questions.
        """
        print("---DECIDE PLAN---")
        # Get the plan questions from the state
        questions = state["questions"]
        print("Questions in decide plan", questions)
        # Decide action according to good questions threshold.
        if len(questions) > self.max_queries * self.threshold:
            return "vector_search"
        else:
            return "plan"

    def vector_search(self, state: GraphState) -> GraphState:
        """
        Perform a vector search using the Chroma vector store.
        """
        print("---VECTOR SEARCH---")
        questions = state["questions"]
        # Initialize an empty context to hold the search results
        contexts = []
        # Loop through each generated search query
        for query in questions:
            # Perform a search using the Chroma vector store
            search_result = CustomDataChat().store.similarity_search(query)
            # Extract the contents from the search results.
            print("Search result", search_result)
            result = '\n'.join([doc.page_content for doc in search_result])
            print("res", result)
            # Append the extracted content to the context.
            contexts.append(result)
            sleep(10)

        print({"questions in vector search": questions, "contexts in vector search": contexts})
        return {"questions": questions, "contexts": contexts}

    def grade_contexts(self, state: GraphState) -> GraphState:
        print("---GRADE CONTEXTS---")
        questions = state["questions"]
        contexts = state["contexts"]

        prompt = prompts.review_context_template()
        good_contexts = []
        chain = prompt | self.llms["grade"] | StrOutputParser()

        for index in range(len(questions)):
            # Invoke the chain with the current question and context
            result = chain.invoke({"questions": questions[index], "contexts": contexts[index]})
            print(f"Result: {result} question: {questions[index]}\n")

            # Parse the JSON content
            scores = json.loads(result).get("score", "")
            print("scores in grade context", scores)
            # Check the score for the current question
            question = questions[index]
            print("question for grading is", question)
            if scores == "yes":
                # Only keep good contexts
                good_contexts.append(contexts[index])
            sleep(1)
        print("good contexts are", good_contexts)
        state["contexts"] = good_contexts
        return state

    def decide_contexts(self, state: GraphState) -> str:
        """
        Decide if contexts is good enough.
        """
        print("---DECIDE CONTEXTS---")
        # Get the contexts from the state
        contexts = state["contexts"]

        # Decide action according to good questions threshold.
        if len(contexts) > self.max_queries * self.threshold:
            return "useful"
        else:
            return "not useful"

    def generate_report(self, state: GraphState) -> GraphState:
        """
        Generate a report based on the contexts.
        """
        print("---GENERATE REPORT---")
        context = state["contexts"]
        topic = state["topic"]

        # Join all of the context strings together into a single string.
        context = "\n".join(context)
        # print(f"context: {context}")

        # Generate a report template based on the given topic and context
        prompt = prompts.report_template()

        # Call the LLM with the generated review template to get a response
        chain = prompt | self.llms["text"] | StrOutputParser()
        report = chain.invoke({"topic": topic, "context": context})
        print(f"report: {report}")

        return {"report": report, "contexts": context}

    def review_report(self, state: GraphState) -> str:
        """
        Review report and decide whether to pass or fail.
        """
        print("---REVIEW REPORT---")
        # Get the plan questions from the state
        report = state["report"]
        # Get the topic from the state
        contexts = state["contexts"]

        # Generate a review template based on the given topic and questions
        prompt = prompts.review_report_template()

        # Call the LLM with the generated review template to get a response
        chain = prompt | self.llms["grade"] | StrOutputParser()
        result = chain.invoke({"contexts": contexts, "report": report})
        print(f"Result: {result}\n")

        # Generate result.
        score = json.loads(result).get("score", "")
        return "pass" if score == "yes" else "fail"

    def save(self, state: GraphState) -> dict:
        """
        Save result to files.
        """
        print("---SAVE---")
        context = state["contexts"]
        report = state["report"]

        with open("context.txt", "w", encoding='utf-8') as f:
            f.write(context)
        with open("report.txt", "w", encoding='utf-8') as f:
            f.write(report)

        return {"report": report}

    def build_graph(self):
        """
        Build graph for literature review.
        """
        # Create graph
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("plan", self.plan)
        workflow.add_node("review_plan", self.review_plan)
        workflow.add_node("vector_search", self.vector_search)
        workflow.add_node("grade_contexts", self.grade_contexts)
        workflow.add_node("generate_report", self.generate_report)
        workflow.add_node("save", self.save)

        # Build graph
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "review_plan")
        workflow.add_conditional_edges(
            "review_plan",
            self.decide_plan,
            {
                "plan": "plan",
                "vector_search": "vector_search",
            },
        )
        workflow.add_edge("vector_search", "grade_contexts")
        workflow.add_conditional_edges(
            "grade_contexts",
            self.decide_contexts,
            {
                "useful": "generate_report",
                "not useful": "vector_search",
            },
        )
        workflow.add_conditional_edges(
            "generate_report",
            self.review_report,
            {
                "pass": "save",
                "fail": "plan",
            },
        )
        workflow.add_edge("save", END)
        return workflow

    def review(self, topic: str) -> None:
        """
        Generates a review for a given topic.
        """
        # Build graph
        workflow = self.build_graph()
        print("Generate graph")
        # Compile the graph
        app = workflow.compile()
        print("Compile graph")
        # Run the graph
        print("Run graph")
        inputs = {"topic": topic, "max_queries": self.max_queries}
        report_content = None
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Finished running: {key}")
                if key == "save":
                    report_content = value["report"]
        print("Finish")
        return report_content


if __name__ == "__main__":
    topic = "What are the test specifications for Compliant Device Test?"
    reviewer = LiteratureReviewer()
    reviewer.review(topic)