from datetime import datetime
from langchain_core.prompts import PromptTemplate


def plan_template() -> PromptTemplate:
    """
    Generates the search queries prompt for the given topic.
    Returns: PromptTemplate: The search queries prompt for the given topic.
    """
    plan_prompt = PromptTemplate(
        input_variables=["max_queries", "topic"],
        template="""Write {max_queries} queries to search
        that form an objective opinion from the following task: "{topic}"
        Use the current date if needed: """ + f'{datetime.now().strftime("%B %d, %Y")}.' + """Also
        include in the queries specified task details such as locations, names, etc.
        You must respond with the following json format:
        key is an integer, value is the question".""",)
    return plan_prompt


def review_plan_template() -> PromptTemplate:
    """
    Generates the review queries prompt for the given topic.
    Returns: PromptTemplate: The review queries prompt for the given topic.
    """
    review_plan_prompt = PromptTemplate(
        input_variables=["topic", "questions"],
        template="""Verify to resolve the topic: {topic}, and following questions:
        {questions} \n
        Give a binary score 'yes' or 'no' score to each question to indicate
        whether the question is relevant to the topic. \n
        Provide the binary score as a JSON with a single key 'score' and
        no premable or explaination.""",)
    return review_plan_prompt


def review_context_template() -> PromptTemplate:
    """
    Generates the review context prompt for the given topic.
    Returns: PromptTemplate: The review context prompt for the given questions.
    """
    review_prompt = PromptTemplate(
        input_variables=["questions", "contexts"],
        template="""Verify to resolve the questions: {questions}, and following search result:
        {contexts} \n
        Give a binary score 'yes' or 'no' score to indicate whether the serach result is
        relevant to the topic.
        Provide the binary score as a JSON with a single key 'score' and
        no premable or explaination.""",)
    return review_prompt


def report_template() -> PromptTemplate:
    """
    Generate the template for generating the final report.
    Returns: PromptTemplate: The final report template.
    """
    report_prompt = PromptTemplate(
        input_variables=["topic", "context"],
        template="""Verify to resolve the topic: {topic}, and following context:
        {context} \n
        Generate a report only using the information from context and the defined topic.""",)
    return report_prompt


def review_report_template() -> PromptTemplate:
    """
    Generates the review report prompt for the given report to see if it matches the contexts.
    Returns: PromptTemplate: The review context prompt for the given questions.
    """
    review_prompt = PromptTemplate(
        input_variables=["report", "contexts"],
        template="""
        Here is the report: {report} \n
        Here are the contexts: {contexts}. \n
        Give a binary score 'yes' or 'no' score to indicate whether the report covers the
        contexts well. \n
        Provide the binary score as a JSON with a single key 'score' and no premable
        or explaination.""",)
    return review_prompt
