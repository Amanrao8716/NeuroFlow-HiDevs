.PHONY: security-scan

security-scan:
	docker build --target runtime -t neuroflow-api:task15-prod -f backend/Dockerfile .
	trivy image --severity CRITICAL --exit-code 1 neuroflow-api:task15-prod
