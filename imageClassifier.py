import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-08-06-05-56-12-714"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a Predictor
    predictor = Predictor(
        endpoint_name=ENDPOINT,
        sagemaker_session=sagemaker.Session()
    )

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    # event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': {
            "image_data": event["body"]["image_data"],
            "s3_bucket": event["body"]["s3_bucket"],
            "s3_key": event["body"]["s3_key"],
            "inferences": inferences
        }
    }