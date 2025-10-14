# Spark Data Agent

## About The Project

Spark Data Agent is an interactive platform designed for the analysis of large volumes of textual data, specifically online posts, to identify and study linguistic patterns associated with depression. The project leverages Big Data and Artificial Intelligence technologies to make complex data analysis accessible through an intuitive conversational interface.

The core of the platform is built on three technological pillars:
* **Apache Spark**: For efficient, distributed processing of large datasets.
* **Conversational AI**: An advanced agent orchestrated by LlamaIndex and powered by the Gemini 2.5 Flash Large Language Model, allowing users to ask questions in natural language.
* **Web Interface**: A user-friendly and reactive web application built with Streamlit.

This system empowers researchers and analysts, who may not have programming expertise, to perform complex lexical, frequency, temporal, and behavioral analyses in real-time.

## Key Objectives

The project was guided by four primary goals:
1.  **Robust Data Pipeline**: Develop an automated process to transform raw XML data into the highly optimized Parquet format to ensure high-performance interactive analysis.
2.  **Advanced Analysis Toolkit**: Implement a suite of statistical and linguistic analysis functions using PySpark, each encapsulated as a standalone "tool".
3.  **Conversational AI Agent**: Integrate an LLM to understand user queries, select the appropriate PySpark analysis tool, execute it with the correct parameters, and interpret the results.
4.  **Intuitive Web Interface**: Create a web application using Streamlit that allows users to interact with the AI agent via a chat interface and view results in clear formats like tables and graphs.

## Architecture Overview

The system architecture is designed to be modular and scalable, ensuring a logical data flow from the raw source to the end-user.

**Conceptual Flow:**
1.  **Offline Pre-processing**: Raw data (XML files and a `truth.txt` file for classification) is processed by a Python script (`preprocess.py`). This script cleans, structures, and enriches the data, saving it in the optimized Parquet format.
2.  **User Interaction**: The user interacts with the Streamlit web application.
3.  **AI Agent**: User requests are interpreted by the AI agent, which is orchestrated by LlamaIndex and powered by Gemini. The agent selects the appropriate PySpark tool based on the user's query.
4.  **Data Processing**: A central `SparkDataManager` class manages the Apache Spark session and data loading. It loads the Parquet data into Spark DataFrames for analysis.
5.  **Execution & Response**: The selected PySpark function is executed by Spark on the data. The results are returned to the agent, which then formats and presents them to the user in a clear and interpretable way.

## Core Technologies

* **Backend and Data Processing**: **Apache Spark** (via PySpark) is used for its ability to efficiently handle terabyte-scale datasets through distributed, in-memory processing.
* **Artificial Intelligence**: **LlamaIndex** orchestrates the agent's logic, transforming PySpark analysis functions into `FunctionTool` objects that the LLM can use. The **Gemini 2.5 Flash** LLM understands user intent and selects the appropriate tools.
* **Frontend**: **Streamlit** was chosen for its ability to rapidly create interactive web interfaces for data science applications.

## Analysis Capabilities

The platform offers a range of analytical tools available to the AI agent:

* **Verbosity Analysis**: Compares the length of posts and titles between different user groups (e.g., depressed vs. control) to identify behavioral indicators.
* **Lexical and Frequency Analysis**: Identifies and compares the most frequently used words by each group to uncover differences in vocabulary. The process involves tokenization, normalization, and stopword removal.
* **Temporal Analysis**: Explores posting patterns over time, such as activity by hour of the day or day of the week, using Spark's temporal functions.
* **Behavioral Analysis**: Quantifies user posting behavior, including the total number of posts and the frequency of contributions.
* **Keyword Search**: Allows researchers to perform targeted thematic analyses by filtering posts containing specific terms (e.g., "anxiety," "loneliness") using regular expressions in Spark.

## The Conversational Agent

The AI agent's behavior is governed by a detailed system prompt that ensures every interaction is methodical, transparent, and scientifically rigorous. Key aspects of its design include:

* **Defined Persona**: Acts as an expert AI assistant specialized in text data analysis with Apache Spark for depression studies.
* **Mandatory Workflow**: Follows a strict sequence for each request: presents an action plan, verifies data status, declares the tool and parameters it will use, and requests confirmation for complex analyses.
* **Structured Responses**: Outputs are formatted in Markdown, including an executive summary, the action plan, raw results, an interpretation, and a critical analysis of limitations (e.g., "correlation does not imply causation").

## Future Work

The project lays the groundwork for several future enhancements:

* **Advanced NLP Analysis**: Integrate more sophisticated techniques like Topic Modeling (LDA) and Sentiment Analysis.
* **Interactive Visualizations**: Enhance the interface with dynamic and complex charts generated by the agent using libraries like Plotly or Altair.
* **Real-Time Data Pipeline**: Evolve the system to a streaming architecture using Spark Streaming or Kafka to analyze data as it is generated.
* **User-Specific Customization**: Allow researchers to upload their own datasets and labels to use the platform for their specific research needs.