import re
import colorama
from transformers import pipeline

with open("context.txt", "r") as file:
    context = re.sub(r"\s+", " ", " ".join([line.strip() for line in file.readlines()]))

sentenceEndPoints = []

for sentence in context.split("."):
    if len(sentenceEndPoints) == 0:
        sentenceEndPoints.append(len(sentence) + 1)
    else:
        sentenceEndPoints.append(sentenceEndPoints[-1] + len(sentence) + 1)

colorama.init()

MODEL_NAME = "deepset/roberta-base-squad2"

nlp = pipeline("question-answering", model=MODEL_NAME, tokenizer=MODEL_NAME)

question = input("Q: ")

while question:
    QA_input = {
        "question": question,
        "context": context
    }

    result = nlp(QA_input)

    start: int = result["start"]
    end: int = result["end"]

    sentenceStart = 0
    for sentenceEndPoint in sentenceEndPoints:
        if sentenceEndPoint > start:
            break

        sentenceStart = sentenceEndPoint

    sentenceEnd = 0
    for sentenceEndPoint in sentenceEndPoints:
        if sentenceEndPoint > end:
            sentenceEnd = sentenceEndPoint
            break

    answer: str = result["answer"].strip()

    print("A: ", context[sentenceStart:start].strip(), colorama.Fore.RED, answer, colorama.Style.RESET_ALL, context[end:sentenceEnd].strip())

    question = input("Q: ")
