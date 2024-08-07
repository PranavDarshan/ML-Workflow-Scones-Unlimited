# ML-Workflow-Scones_Unlimited

This project was given by Udacity for their Fundamentals of Machine Learning Course. 

The problem statement given by Udacity was:

You are hired as a Machine Learning Engineer for a scone-delivery-focused logistics company, Scones Unlimited, and you’re working to ship an Image Classification model. The image classification model can help the team in a variety of ways in their operating environment: detecting people and vehicles in video feeds from roadways, better support routing for their engagement on social media, detecting defects in their scones, and many more!

In this project, you'll be building an image classification model that can automatically detect which kind of vehicle delivery drivers have, in order to route them to the correct loading bay and orders. Assigning delivery professionals who have a bicycle to nearby orders and giving motorcyclists orders that are farther can help Scones Unlimited optimize their operations.

As an MLE, your goal is to ship a scalable and safe model. Once your model becomes available to other teams on-demand, it’s important that your model can scale to meet demand, and that safeguards are in place to monitor and control for drift or degraded performance.

In this project, you’ll use AWS Sagemaker to build an image classification model that can tell bicycles apart from motorcycles. You'll deploy your model, use AWS Lambda functions to build supporting services, and AWS Step Functions to compose your model and services into an event-driven application. At the end of this project, you will have created a portfolio-ready demo that showcases your ability to build and compose scalable, ML-enabled, AWS applications.

## Step 1: Data Staging

In this step, the data is extracted from the [CIFAR dataset hosting service](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) using get requests and then finally processing and visualizing the data. The data is then converted to images and stored in the test and the train folder. Then these folder contents are pushed to the corresponding S3 bucket of the SageMaker instance. 

#### Note: The data containing only the bicycle and motorcycle are considered.

## Step 2: Model Training and Deployment

The function image_uris is used to retreive a model and create a training job
```ruby
algo_image = retrieve('image-classification', region=region, version='latest')
```

Create an estimator for training job

```ruby
img_classifier_model=sagemaker.estimator.Estimator(image_uri=algo_image,
                                                   role=role,
                                                   instance_count=1,
                                                   instance_type='ml.p3.2xlarge',
                                                   output_path=s3_output_location,
                                                   sagemaker_session=sagemaker_session
)
```
Set the hyperparameters for the estimator: 

```ruby
img_classifier_model.set_hyperparameters(
    image_shape='3,32,32', 
    num_classes=2, 
    num_training_samples=1000
)
```
Training job is created by 

```ruby
img_classifier_model.fit(model_inputs)
```

The instance type for deployment can be changed. Here 'ml.m5.xlarge' is used. Finally deploy the model using :

```ruby
deployment = img_classifier_model.deploy(
    # fill in deployment options
    initial_instance_count=1,
    instance_type='ml.m5.xlarge', 
    data_capture_config=data_capture_config
    )

endpoint = deployment.endpoint_name
print(endpoint)
```
After deployment, create a predictor and predict the class of an inference.

## Step 4: Create 3 Lambda Functions and a Step Function

The code for all the lambda functions are specified and remember to include sagemaker libraries before updating the lmabda functions. The Lambda functions can be tested by giving a test event:

This also used as payload for the state machine.

```ruby
{
    "inferences": [], # Output of predictor.predict
    "s3_key": "", # Source data S3 key
    "s3_bucket": "", # Source data S3 bucket
    "image_data": ""  # base64 encoded string containing the image data
}
```
The test event must execute successfully. The log can be found in the [image_classifier_lambda.txt](https://github.com/PranavDarshan/ML-Workflow-Scones_Unlimited/blob/main/image_classifier_lambda.txt). This is the output passed to the next state.

<img src=https://github.com/PranavDarshan/ML-Workflow-Scones_Unlimited/blob/main/assets/imageClassfier_lambda.png/>

Create a state machine with the architecture:

<img src=https://github.com/PranavDarshan/ML-Workflow-Scones_Unlimited/blob/main/assets/stateMachineArchitecture.png>

The state machine must execute successfully when the payload is given as the same input text.

<img src=https://github.com/PranavDarshan/ML-Workflow-Scones_Unlimited/blob/main/assets/state_machine.png>

## Step 5: Testing and Evaluation

You will first perform several step function invokations using data from the test dataset. This process should give you confidence that the workflow both succeeds AND fails as expected. In addition, you will use the captured data from SageMaker Model Monitor to create a visualization to monitor the model.
The model is monitored and visualized by plotting graphs and histograms.
