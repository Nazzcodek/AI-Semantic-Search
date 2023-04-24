import openai
import os
import pinecone
from tqdm.auto import tqdm
from datasets import load_dataset

openai.api_key = os.environ['OPENAI_API_KEY']

def encode_text(text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Encode the following text as a vector:\n\"{text}\"",
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.0,
    )

    return response.choices[0].text.strip()

dataset = load_dataset('quora', split='train')
dataset[:5]

questions = []

for record in dataset['questions']:
    questions.extend(record['text'])
  
# remove duplicates
questions = list(set(questions))
print('\n'.join(questions[:5]))
print(len(questions))

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=('asia-northeast1-gcp')
)

index_name = 'semantic-search-fast'

# only create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=2048,  # GPT-3's embedding size
        metric='cosine'
    )

# now connect to the index
index = pinecone.GRPCIndex(index_name)

batch_size = 128

for i in tqdm(range(0, len(questions), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(questions))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadatas = [{'text': text} for text in questions[i:i_end]]
    # create embeddings
    xc = [encode_text(text) for text in questions[i:i_end]]
    # create records list for upsert
    records = zip(ids, xc, metadatas)
    # upsert to Pinecone
    index.upsert(vectors=records)

# check number of records in the index
index.describe_index_stats()

query = "which city has the highest population in the world?"

# create the query vector
xq = encode_text(query)

# now query
xc = index.query(xq, top_k=3, include_metadata=True)
xc

for result in xc['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")

pinecone.delete_index(index_name)
