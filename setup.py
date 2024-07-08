from setuptools import setup, find_packages

setup(
    name='roboeval',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "pandas==2.0.3",
        "text_generation==0.6.1",
        "transformers==4.42.3",
        "google-generativeai",
        "openai==0.27.8",
        "websockets==12.0"
        ],
    author='Zichao Hu',
    author_email='zichao@utexas.edu',
    description='Adapted RoboEval Implementation',
    url='https://github.com/ut-amrl/codebotler',
)
