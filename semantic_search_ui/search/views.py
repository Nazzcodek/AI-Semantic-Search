from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import openai
import pinecone
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

model_engine = "text-davinci-002"
prompt = (
    "Find the most similar text to the given query using OpenAI's " 
    "text-davinci-002 model."
)

index_name = "semantic-search-fast"
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], 
    environment="asia-northeast1-gcp")

index = pinecone.Index(index_name)


@csrf_exempt
def search(request):
    query = ""
    results = []

    if request.method == "POST":
        query = request.POST.get("query")

        # generate the query using OpenAI
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=64,
            n=1,
            stop=None,
            temperature=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            completion_context={"query": query},
        )
        generated_query = response.choices[0].text.strip()

        # query the index
        xq = [generated_query]
        xc = index.query(xq, top_k=3, include_metadata=True)

        for result in xc["matches"]:
            score = round(result["score"], 2)
            text = result["metadata"]["text"]
            results.append({"score": score, "text": text})

    context = {"query": query, "results": results}
    return render(request, "search/search.html", context)
