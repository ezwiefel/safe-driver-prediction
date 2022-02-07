environment:
	az ml environment create --file ./safe-driver/env/environment.yml

train:
	az ml job create --file ./safe-driver/src/train/job.yml

train-pipeline:
	az ml job create --file ./safe-driver/src/training-pipeline.yml