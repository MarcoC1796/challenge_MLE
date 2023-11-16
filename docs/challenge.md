# Flight Delay Prediction Project Report

## Executive Summary

This report documents the process, analysis, and decisions taken during the implementation of a flight delay prediction model for SCL airport. The project involves model selection and justification, code transposition from a Jupyter notebook to a production-quality API, and deployment using cloud services.

## Contents

- [Introduction](#introduction)
- [Model Selection Rationale](#model-selection-rationale)
- [Model Implementation Details](#model-implementation-details)
- [API Development](#api-development)
- [Deployment Strategy](#deployment-strategy)
- [CI/CD Pipeline](#cicd-pipeline)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)

## Introduction

The provided Jupyter Notebook, `exploration.ipynb`, contained the initial work of a Data Scientist (DS) which included data exploration and preliminary model training. The task was to operationalize this model to predict the probability of delay for flights associated with SCL airport.

## Model Selection Rationale

### Final Decision

The Logistic Regression model was selected for deployment for the following reasons:

- **Interpretability**: Vital for stakeholders to understand the factors influencing delay predictions.
- **Speed**: Logistic Regression is computationally less intensive, allowing for quick retraining and predictions.
- **Operational Alignment**: Fits the need for a model that can be frequently updated with new data.

### Trade-offs

Choosing Logistic Regression involved trade-offs:

- **Predictive Performance**: Potentially less capable of capturing complex patterns compared to models like XGBoost.
- **Class Imbalance**: Logistic Regression may not handle class imbalances as effectively as other algorithms.

### Future Considerations

Continuous monitoring of model performance is recommended. Adjustments and re-evaluations may be required as new data comes in and business needs evolve.

## Model Implementation Details

The model was transcribed from the `.ipynb` file into `model.py`. During this process, the code was refactored to adhere to best programming practices. Notable implementations include:

- **Class Weight Balancing**: Implemented to handle class imbalance in the dataset.
- **Top 10 Features**: The model was restricted to use only the top 10 features as per the DS's analysis.
- **Error Handling**: Improved the robustness of the model with better error handling and input validation.

## API Development

The model was deployed as a REST API using FastAPI. The API endpoints were designed to receive flight data and return delay predictions. The code in `api.py` is structured to handle requests efficiently and return predictions in a standardized format.

## Deployment Strategy

The API was containerized using Docker and deployed on Google Cloud Platform (GCP) via Cloud Run. This choice was motivated by the simplicity and scalability offered by GCP.

## CI/CD Pipeline

GitHub Actions was used to set up the CI/CD pipeline. The pipeline includes:

- **Continuous Integration**: Automated linting and tests to ensure code quality and functionality.
- **Continuous Deployment**: Automated deployment to Cloud Run upon successful integration, ensuring the latest version of the API is always available.

## Challenges and Solutions

During the project, several challenges were encountered:

- **Data Preprocessing**: Ensuring the preprocessing in `model.py` matched the Jupyter notebook.
- **Category Mapping**: Dealing with unseen categories in incoming API requests.

Solutions implemented:

- **Refactoring**: Streamlined preprocessing code to align with the notebook.
- **Validation**: Improved category validation in the API to handle unknown categories gracefully.

## Future Enhancements

Potential future enhancements include:

- **Model Monitoring**: Implement monitoring to track model performance over time.
- **Feature Engineering**: Explore additional features that could improve model accuracy.
- **Auto-Scaling**: Optimize cloud resources for cost and performance.

## Conclusion

The successful completion of this challenge demonstrates the capability to transform a data science prototype into a production-ready service. The choices made throughout were aimed at creating a reliable, interpretable, and maintainable system that meets the operational needs of the airport team.

---

_This report was prepared by Marco Chacón, and documents the work done for the Software Engineer (ML & LLMs) Challenge._
