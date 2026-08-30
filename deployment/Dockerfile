FROM public.ecr.aws/lambda/python:3.14

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

COPY heart_disease_pipeline.joblib ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
