import boto3
import json
import time

class AWSDeployer:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.ecs = boto3.client('ecs', region_name=region)

    def create_gpu_instance(self, instance_type='g5.xlarge', key_name='qythera'):
        response = self.ec2.run_instances(
            ImageId='ami-0c55b159cbfafe1f0',
            InstanceType=instance_type,
            KeyName=key_name,
            MinCount=1, MaxCount=1,
            SecurityGroupIds=['sg-xxxxxxxx'],
            UserData=self._get_userdata(),
            TagSpecifications=[{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'qythera-gpu'}]}],
        )
        instance_id = response['Instances'][0]['InstanceId']
        print(f'Created instance: {instance_id}')
        return instance_id

    def _get_userdata(self):
        return '''#!/bin/bash
apt-get update && apt-get install -y python3-pip docker.io
pip3 install qythera
docker pull qythera:latest
docker run -d --gpus all -p 8000:8000 qythera:latest
'''

    def create_s3_bucket(self, bucket_name):
        s3 = boto3.client('s3', region_name=self.region)
        s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': self.region})
        print(f'Created S3 bucket: {bucket_name}')

    def setup_cloudwatch(self, namespace='Qythera'):
        cw = boto3.client('cloudwatch', region_name=self.region)
        cw.put_metric_alarm(
            AlarmName='HighGPUUsage', MetricName='GPUUtilization',
            Namespace=namespace, Statistic='Average', Period=300,
            EvaluationPeriods=2, Threshold=90, ComparisonOperator='GreaterThanThreshold',
        )
