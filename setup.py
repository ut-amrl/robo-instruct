from setuptools import setup, find_packages

setup(
    name='robo-instruct',
    version='0.1',
    packages=find_packages(['roboeval', 'robo_instruct']),
    install_requires=[
        "pandas==2.0.3",
        "text_generation==0.6.1",
        "transformers==4.42.3",
        "google-generativeai",
        "openai==0.27.8",
        "websockets==12.0",
        "vllm==0.5.1",
        "numpy==1.24.4",
        "joblib==1.3.2",
        "torch==2.3.0",
        "levenshtein"
    ],
    author='Zichao Hu',
    author_email='zichao@utexas.edu',
    description='Adapted RoboEval Implementation',
    url='https://github.com/ut-amrl/codebotler',
)