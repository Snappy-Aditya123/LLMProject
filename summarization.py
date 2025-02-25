from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
import DBMS  # Your database module


class TextSummarizer:
    def __init__(self, model="granite3.1-dense:2b"):
        self.llm = ChatOllama(model=model, temperature=0.7, keep_alive=False, num_thread=6)

    def summarize_text(self, text):
        summary_prompt = f"Summarize the following text concisely retaining major information:\n{text}"
        summary = self.llm.invoke([HumanMessage(summary_prompt)]).content
        return summary

    def summarize_conversation(self, messages):
        summary_prompt = f"Summarize the following conversation concisely retaining major information like names, ages, numbers, and important parts only:\n{messages}"
        summary = self.llm.invoke([HumanMessage(summary_prompt)]).content
        return summary

    def summarize_cv(self, cv_text):
        summary_prompt = f"Summarize the following CV concisely retaining major information:\n{cv_text}"
        summary = self.llm.invoke([HumanMessage(summary_prompt)]).content
        return summary

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import json

bot = ChatOllama(model='g-career-assistant')
message = [
    SystemMessage(
    "Generate output in JSON format based on the user's input. The JSON should include: "
    "'search' (Boolean) — set to true if searching for available jobs is useful based on the user's query, otherwise false;(true if the user mentions a job title like softwarte engineer, teacher etc ) Try to use this only if find it important "
    "'keyword' (list) — relevant keyword extracted from the user's input to perform the job search (leave empty if 'search' is false)(only accepts one job title); "
    "and 'response' (string) — the bot's natural language reply to the user. "
    "Use the search functionality only when it would add value and directly answer the user's query"
    "The Keyowrds are always job titles, so if the user mentions a job title, then the search should be done."
),
HumanMessage("please tell if this is a good carrier adive I want to be a english teacher for me A high-demand job typically means that either there are more opportunities than qualified candidates to fill them, or that there will be more jobs available over the next several years. These types of jobs tend to have several benefits, including more competitive salaries, increased opportunities for advancement, and in some instances, even greater job security.  "),
]
z = bot.invoke(message)

def parser(z):
    response = z.content
    json_resp = json.loads(response)
    search = json_resp['search']
    keyword = json_resp['keyword']
    response = json_resp['response']
    return search, keyword, response

print(parser(z))