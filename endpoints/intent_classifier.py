import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from configurations.utils import chunk_data, extract_intent
from langchain.chat_models import ChatOpenAI
import openai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



class IntentClassifier:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        openai.api_key = api_key

    def is_ready(self) -> bool:
        return True

    def create_templates(self, data, chunk_size):
        """
            Generate templates using the provided data.

            Args:
                data (dict): A dictionary containing data for template generation.
                chunk_size: The size of chunks to generate.

            Returns:
                list: A list of template strings generated from the data.

            This function creates template strings by substituting placeholders in a base template
            with corresponding values from the provided data dictionary. It returns a list of
            generated template strings formated in langchain.templates.
            """
        # intents = extract_intents(train_data)

        def create_few_shot_prompt_template(chunk):
            example_template = PromptTemplate(
                input_variables=["intent", "input"],
                template="Intent: {intent}\nExample: {input}"
            )

            few_shot_prompt_template = FewShotPromptTemplate(
                examples=chunk,
                example_prompt=example_template,
                prefix="The following are examples of various intents related to Airline Travel Information Systems:\n",
                suffix="\nBased on the above examples, classify the following input:\nInput: {input}\nIntent:",
                input_variables=["input"],
            )

            return few_shot_prompt_template

        chunks = list(chunk_data(data, chunk_size))
        chunked_templates = [create_few_shot_prompt_template(chunk) for chunk in chunks]
        return chunked_templates

    def classify_inputs(self, input_text, chunked_templates):
        """
           Classify input text using chunked templates.

           Args:
               input_text (str): The input text to classify.
               chunked_templates (list): A list of template chunks.

           Returns:
               str: The final response after classification.

           This method classifies the input text by running it through each template chunk
           and aggregating the responses. It uses the provided input text and a list of
           template chunks to perform classification.
           """
        start_time = time.time()
        llm = ChatOpenAI(model_name=self.model_name, openai_api_key=openai.api_key, temperature=0)


        # Function to run classification on a chunk

        def run_on_chunk(template):
            chain = LLMChain(llm=llm, prompt=template)
            return chain.run(input=input_text, max_tokens=10)

        with ThreadPoolExecutor() as executor:

            future_to_template = {executor.submit(run_on_chunk, template): template for template in chunked_templates}
            responses = []

            # Collect results as they become available
            for future in as_completed(future_to_template):
                response = future.result()
                responses.append(response)

        intent_labels = [extract_intent(response) for response in responses]
        intent_counts = Counter(intent_labels)
        top_3_predictions = [{"label": label} for label, _ in intent_counts.most_common(3)]

        end_time = time.time()
        print(f"Processed Intent in {end_time - start_time} seconds.")
        return top_3_predictions

    def evaluate_model(self, test_data, chunked_templates):
        """
           Evaluate the performance of the model using test data.

           Args:
               test_data (list): A list of dictionaries containing test data, where each dictionary
                                 represents a test instance with 'input' and 'intent' keys.
               chunked_templates (list): A list of template chunks.

           Returns:
               dict: A dictionary containing evaluation metrics including accuracy, precision, recall, and F1 score.

           This method evaluates the performance of the model by comparing its predictions
           on the provided test data against the true labels. It calculates accuracy, precision,
           recall, and F1 score as evaluation metrics.
        """

        y_true = [item['intent'] for item in test_data[:300]]
        pred_list = []
        y_pred = []
        for item in test_data[:300]:
            input_text = item['input']
            prediction = self.classify_inputs(input_text, chunked_templates)
            pred_list.append(prediction)

        for sublist in pred_list:
            labels = [item['label'] for item in sublist]
            y_pred.append(','.join(labels))

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

