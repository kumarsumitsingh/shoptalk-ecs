# shoptalk-ecs
This project is deployed to AWS ECS. All the images are registered to ECR.
To run this project,you need to 

1 CLI to aws using command:
    aws config

2 Logging in to ECR:
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWSAccoundID>.dkr.ecr.us-east-1.amazonaws.com

3 Build the image 
docker build -t shoptalk-bot . 
Note: Please make sure your local docker is running

4 Tag the image in ECR
docker tag shoptalk-bot:latest <AWSAccoundID>.dkr.ecr.us-east-1.amazonaws.com/capstone/shoptalk-bot

Step-5 Push the image to ECR
docker push <AWSAccoundID>.dkr.ecr.us-east-1.amazonaws.com/capstone/shoptalk-bot:latest