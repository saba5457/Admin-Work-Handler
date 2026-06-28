from flask import Flask, render_template, request, jsonify
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv("../LangChain/.env")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=1000
)

prompts = {
    "tasks": ChatPromptTemplate.from_messages([
        ("system", """You are a professional task manager. Given a list of tasks:
1. Number each task
2. Assign priority: 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
3. Estimate time for each task
4. Create an optimized daily schedule
5. Add a motivational tip at the end
Be specific and professional."""),
        ("human", "{input}")
    ]),
    "email": ChatPromptTemplate.from_messages([
        ("system", """You are a professional email writer. Write a complete, polished email based on the user's request.
Include:
- Subject line
- Professional greeting
- Clear body
- Professional closing
- Signature placeholder
Keep it concise and professional."""),
        ("human", "{input}")
    ]),
    "report": ChatPromptTemplate.from_messages([
        ("system", """You are a professional report writer. Create a well-structured report based on the input.
Include:
- Executive Summary
- Key Points
- Details
- Conclusions
- Recommendations
Format it professionally."""),
        ("human", "{input}")
    ]),
    "meeting": ChatPromptTemplate.from_messages([
        ("system", """You are a professional meeting summarizer. Based on the meeting notes provided:
1. Create a clean summary
2. List key decisions made
3. List action items with owners
4. List next steps
5. Note any important deadlines
Be clear and concise."""),
        ("human", "{input}")
    ])
}

chains = {key: prompt | llm | StrOutputParser() for key, prompt in prompts.items()}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    tool = data.get("tool", "tasks")
    user_input = data.get("input", "")
    
    if not user_input:
        return jsonify({"result": "Please provide some input!"})
    
    result = chains[tool].invoke({"input": user_input})
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True, port=5002)