# chatbot_flask

## Purpose 

This is a simple APP made with Groq, Sreamlit and LLAMA model. Streamlit can be used to make a fast prototyping tool for machine learning applciations. It can be easily converted to React and Vue apps as state management and redering is similar. 
This is a simple chatbot made thorough groq and llama model with memory. This is to test fast inference speed of this combination.

## Techstack

Streamlit - Frontend
Used for simplicilty and ease of use

Inference - Groq
Used for querying the model

Model - LLAMA
Used for generating responses

![alt text](<Screenshot 2024-05-07 at 4.03.48 PM.png>)

LLama with groq gives us chespest, fastest and most efficient way to query the model.

## How to run the app

### Step 1 

export your groq token as an environment variable

```bash
export GROQ_TOKEN=your_groq_token
```

### Step 2

Pip install requirements in a python environment
    
    ```bash
    pip install groq
    ```

### Step 3 

run 
    
```bash
streamlit run app.py
```

## Some important concepts that translates to all models

Memory 
We use buffer memory. Similar to a bucket that holds few messages before forgetting.
We do this by using lanchain conversation memory.

Langchain modules are used which gives us the flexiblity to to adapt. For example to test langchain models in this


## To include hugging face models in a prototype all like this. 

To convert your code to utilize a Hugging Face model via their API, you'll need to make several modifications, as the integration and API calls are different between Groq and Hugging Face. Below, I'll outline the steps to adapt your current Streamlit app to use a model from Hugging Face.

### Step 1: Obtain a Hugging Face API Token
First, you need to sign up or log in on the Hugging Face website, navigate to your account settings, and create an API token. This token will allow you to make authenticated requests to the Hugging Face model hosting service.

### Step 2: Install Hugging Face Libraries
Ensure you have the `transformers` library installed, which includes the functionality to easily access pre-trained models and utilize them via the `pipeline` function.

```bash
pip install transformers
```

### Step 3: Modify Your Code

1. **Import Required Modules**:
   Replace or add imports from `transformers` to handle model loading and inference.

   ```python
   from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
   ```

2. **Load the Model Using Hugging Face Pipeline**:
   Update the model loading section to use a Hugging Face model. You can select a model suitable for conversation like `gpt-3` or any other available conversational models.

   ```python
   model_name = st.sidebar.selectbox(
       'Choose a model',
       ['gpt-3', 'EleutherAI/gpt-neo-2.7B', 'facebook/blenderbot-small-90M']
   )

   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   chat = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)  # device=0 for GPU
   ```

3. **Adjust the API Call**:
   Replace the Groq API interaction with the Hugging Face `pipeline` call. You'll need to format your input and handle the response accordingly.

   ```python
   if user_question:
       response = chat(user_question + system_prompt, max_length=150)  # Modify parameters as needed
       response_text = response[0]['generated_text']  # Adjust based on your response structure
       message = {'human': user_question, 'AI': response_text}
       st.session_state.chat_history.append(message)
       st.write("Chatbot:", response_text)
   ```

### Step 4: Handling the Conversational Context
With Hugging Face models, you might need to manage the conversational context manually by appending the history of the conversation to each query, depending on how stateful you want your model to be. The `transformers` pipeline does not inherently manage dialogue history, so you'll need to implement this if necessary.

### Optional: Use Hugging Face Spaces
To deploy this Streamlit app, consider using Hugging Face Spaces, which provides an easy way to host your ML models and apps directly from the Hugging Face ecosystem.
