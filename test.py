import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
# Set your OpenAI API key here
openai.api_key = 'sk-proj-oQIGfEFC6ahTqPv8FJ7mT3BlbkFJIRg24vTwEcPhandZ6ztS'  # Replace with your actual API key

def test_api_key():
    try:
        # Create an instance of the ChatOpenAI class
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai.api_key, temperature=0)

        # Create a simple prompt template
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="You are a helpful assistant. Answer the following question:\n{input}"
        )

        # Create a chain with the LLM and the prompt template
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # Test input
        input_text = "What is the capital of France?"

        # Run the chain with the test input
        response = chain.run(input=input_text)

        # Print the response from the model
        print("API Key works! Here is a response from the model:")
        print(response)
    except Exception as e:
        print(f"API Key test failed with error: {e}")

if __name__ == "__main__":
    test_api_key()
