import pickle
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from utiils import read_data, chunk_data, extract_intent
from langchain.chat_models import ChatOpenAI
import openai


class FineTuneModel:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        openai.api_key = api_key

    def create_templates(self, data, chunk_size=100):
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
        print(f"Processed template in {end_time - start_time} seconds.")
        return top_3_predictions

    # def evaluate_classifier(self, classifier):
    #     test_data = read_data(TEST_DATA_FILE_PATH)
    #
    #     test_data = "\n".join(
    #         [f"#{intent}\n{example}" for intent, examples in test_data.items() for example in examples])
    #
    #     total_examples = len(test_data)
    #     correct_predictions = 0
    #     true_positives = 0
    #     false_positives = 0
    #     false_negatives = 0
    #
    #     for input_text, intent_label in test_data:
    #         predicted_intent = classifier(input_text)
    #         if predicted_intent == intent_label:
    #             correct_predictions += 1
    #             if predicted_intent == intent_label:
    #                 true_positives += 1
    #         else:
    #             false_positives += 1
    #             false_negatives += 1
    #
    #     accuracy = correct_predictions / total_examples
    #     precision = true_positives / (true_positives + false_positives)
    #     recall = true_positives / (true_positives + false_negatives)
    #     f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    #
    #     evaluation_results = {
    #         'accuracy': accuracy,
    #         'precision': precision,
    #         'recall': recall,
    #         'f1_score': f1_score
    #     }
    #
    #     return evaluation_results
