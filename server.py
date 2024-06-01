import os
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from requests import Request
from starlette.responses import JSONResponse

from config import TRAIN_DATA_FILE_PATH, API_KEY, MODEL_NAME
from finetune_model import FineTuneModel
from intent_classifier import IntentClassifier
from utiils import read_data

app = FastAPI()
model = IntentClassifier()


class TextRequest(BaseModel):
    text: str



@app.get('/ready')
def ready():
    """
    Check if the model is ready.

    Returns:
        A JSON response indicating status.
    """

    if model.is_ready():
        return {'status': 'OK'}
    else:
        raise HTTPException(status_code=423, detail='Not ready')


@app.post('/intent')
def intent():
    """
    Endpoint for classifying the intent of a text message.
    Request body should be a json object with a single key 'text' containing the text message to classify.
    Response will be a json object with a single key 'intents' containing a list of labels.
    """
    # Implement this function according to the given API documentation
    pass


@app.post('/finetune')
def finetune_model(request: TextRequest, model_name: str = Query(default=None)):

    # Get Default Value
    if not model_name:
        model_name = MODEL_NAME

    trainmodel = FineTuneModel(model_name, API_KEY)

    train_data = read_data(TRAIN_DATA_FILE_PATH)
    chunked_templates = trainmodel.create_templates(train_data, chunk_size=100)

    res = trainmodel.classify_inputs(request.text, chunked_templates)

    return {"intents": res}


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request, exc: RequestValidationError):

    # Collect all errors related to the body and the "text" field
    body_errors = [err for err in exc.errors() if err['loc'][0] == 'body']
    text_errors = [err for err in exc.errors() if err['loc'] == ('body', 'text')]



    # Determine the response based on the collected errors
    if body_errors and not text_errors:
        # If there are errors indicating the body is missing but no specific "text" field error
        return JSONResponse(
            status_code=400,
            content={"label": "BODY_MISSING", "message": "Request doesn't have a body."}
        )
    elif text_errors:
        # If there are errors indicating the "text" field is missing
        return JSONResponse(
            status_code=400,
            content={"label": "TEXT_MISSING", "message": "\"text\" missing from request body."}
        )

    else:
        # If neither condition is met, return a generic error response
        return JSONResponse(
            status_code=400,
            content={"INTERNAL_ERROR": exc.errors()}
        )


@app.exception_handler(Exception)
def internal_server_error_handler(request, exc):
    # Raise Custom Internal Error message
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "label": "INTERNAL_ERROR",
            "message": str(exc)
        }
    )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')
    args = arg_parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    main()
