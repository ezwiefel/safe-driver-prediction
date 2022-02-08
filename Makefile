environment:
	az ml environment create --file ./safe-driver/env/environment.yml

train:
	az ml job create --file ./safe-driver/src/train/job.yml

debug-pipeline:
	python safe-driver/src/pipeline/pipeline.py --skip-registration --debug

publish-pipeline:
	python safe-driver/src/pipeline/pipeline.py --publish-pipeline