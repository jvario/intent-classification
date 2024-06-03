# Topic-Classification

# I.  Introduction:

#### Project Overview:
The task is to implement a zero or few-shot intent classifier that can be used to provide inferencing service via an HTTP Service. During the classifier development, you should take into consideration the following points: 

  - new set of intents 
  - cost 
  - token size
  - latency
  - and parsable output

#### Dataset:
The dataset is about the Airline Travel Information System (ATIS). ATIS dataset provides large number of messages and their associated intents that can be used in training a classifier. Within a chatbot, intent refers to the goal the customer has in mind when typing in a question or comment. 
- train.tsv : This file includes the train data for the templates
- test.tsv : This file contains the test data for the classifier

You can find the dataset [here](https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem/data).


### Installation


    # go to your home dir
    git clone https://github.com/jvario/intent-classification.git
    cd intent-classifier

    # build the image
    docker build -t my_fast_api .
    docker run -d -p 8080:8080 my_fast_api

  #### api - calls
 - **/intent/?model_name='gpt-3.5-turbo'** :  calling the intent classifier (model_name is optional)
# II.  Pipeline:

#### Intent Classifier:
The techniques used in this implementation include the use of Langchain with chunking to reduce the token size, which helps in managing and processing larger texts more efficiently. Additionally, multi-threading was employed to limit the response time, achieving a response time of around 5 seconds. This combination of techniques not only enhances the model's performance but also ensures efficient and timely processing.

# III.  Results:

| Model         | Sample Size | Accuracy | Precision |
|---------------|-------------|----------|-----------|
| gpt-3.5-turbo | ~300        | 0.86     | 0.85      |


This table represents the evaluation results for modelgpt-3.5-turbo based on dataset. The metrics include Accuracy, Precision.

# IV. Conclusion:
As a result, we can observe that GPT-3.5-turbo achieves an accuracy of approximately 86%, with precision and recall scores of 0.85 and 0.86, respectively. This indicates that GPT-3.5-turbo has very good accuracy. Additionally, its performance is superior in some cases where the intent labels are not well-formatted, making exact matches with the true labels challenging. Moreover, the response time is around 5 seconds, demonstrating its efficiency in processing.## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
