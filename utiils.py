import csv


def read_data(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        intents = []
        for row in reader:
            intents.append({"input": row[0], "intent": row[1]})
    return intents

# # Function to chunk data
def chunk_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def extract_intent(response):
    # Assuming intent is the first word in the response string
    return response.split()[0]