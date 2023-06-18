import os
import json
from django.shortcuts import render
import openai
import numpy as np
import pandas as pd
from django.http import HttpResponse
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from django.views.decorators.csrf import csrf_exempt

openai.api_key = os.environ.get("TOKEN")

# Create your views here.
@csrf_exempt
def prompt(request): # post /api/prompt
    if request.method != 'POST':
        data = {'res': 'Only POST requests allowed!'}
        res = HttpResponse(content=json.dumps(
            data), content_type='application/json')
        return res
    
    df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    df.head()

    data = json.loads(request.body)

    # Access the properties from the JSON data
    prompt = data.get('prompt')

    ans = answer_question(df, question=f"{prompt} Please show me a correct, valid link to a page that also answers my question in a separate paragraph titled Relevant Link(s). Make the links surrounded by a tags with a src attribute loaded with their link.", debug=False)
    output = {'res': ans}
    res = HttpResponse(content=json.dumps(
            output), content_type='application/json')
    return res

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"Sorry, I didn't understand that. Please rephrase.\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""