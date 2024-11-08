from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from IPython.display import display


class DataAnalysis(BaseModel):
    analysis_steps: List[str] = Field(description="Steps to analyze the data")
    cleaning_steps: Optional[List[str]] = Field(description="Steps to clean the data if necessary")
    pandas_code: str = Field(description="Pandas code to perform the analysis")


class DataAnalysisAssistant:
    def __init__(self, file_path, llm_model="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=llm_model)
        self.df = None
        self.original_df = None
        self.metadata = None
        self.file_path = file_path

    def _get_metadata(self, df):
        """Extract meaningful metadata about the dataset"""
        metadata = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'null_counts': df.isnull().sum().to_dict(),
            'sample_values': {col: df[col].sample(min(5, len(df))).tolist() for col in df.columns}
        }
        return metadata

    def load_data(self, data_path):
        """Load data and extract metadata"""
        self.original_df = pd.read_csv(data_path)
        self.df = self.original_df.copy()
        self.metadata = self._get_metadata(self.df)

        print("Dataset loaded successfully!")
        print(f"Shape: {self.metadata['shape']}")
        print("\nColumns:")
        for col in self.metadata['columns']:
            print(f"- {col} ({self.metadata['dtypes'][col]})")

    def _get_analysis_prompt(self):
        return PromptTemplate(
            input_variables=["metadata"],
            template="""
            You are a data scientist analyzing a dataset. Given the following metadata about the dataset:
            {metadata}

            Provide a detailed analysis plan that:
            1. Identifies meaningful columns for analysis (skip ID columns, meaningless aggregations)
            2. Suggests appropriate cleaning steps if necessary
            3. Provides pandas code for initial analysis
            4. The csv is located at 'data/Flight Bookings.csv'

            The response should be detailed but avoid unnecessary steps.

            Respond in this exact JSON format:
            {{
                "analysis_steps": ["step1", "step2", "..."],
                "cleaning_steps": ["clean1", "clean2", "..."],
                "pandas_code": "your pandas code here"
            }}
            """
        )

    def analyze_data(self):
        """Perform initial data analysis"""
        if self.df is None:
            raise ValueError("Please load data first using load_data()")

        parser = PydanticOutputParser(pydantic_object=DataAnalysis)
        prompt = self._get_analysis_prompt()

        # Create the chain using the newer invoke pattern
        chain = prompt | self.llm | parser

        try:
            response = chain.invoke({"metadata": str(self.metadata)})

            # Execute cleaning steps if necessary
            if response.cleaning_steps:
                print("\nExecuting cleaning steps:")
                for step in response.cleaning_steps:
                    print(f"- {step}")
                # Create a local namespace for execution
                local_vars = {"df": self.df}
                exec(response.pandas_code, globals(), local_vars)
                self.df = local_vars["df"]  # Update the dataframe if modified

            # Execute analysis
            print("\nExecuting analysis:")
            result = eval(response.pandas_code, globals(), {"df": self.df})
            display(result)

            return response
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

    def query_data(self, question: str):
        """Interactive query interface"""
        query_prompt = PromptTemplate(
            input_variables=["question", "metadata"],
            template="""
            Given this question about the dataset: {question}

            And these dataset details:
            {metadata}

            Write a pandas code snippet to answer this question and explain what it does.
            Use the variable name 'df' for the dataframe.

            Use this exact format for your response:
            CODE: <your pandas code>
            EXPLANATION: <your explanation>
            """
        )

        # Create the chain using the newer invoke pattern
        chain = query_prompt | self.llm
        response_text = chain.invoke({"question": question, "metadata": str(self.metadata)})

        try:
            # Extract code and explanation
            code = response_text.split("CODE:")[1].split("EXPLANATION:")[0].strip()
            explanation = response_text.split("EXPLANATION:")[1].strip()

            print("Explanation:", explanation)
            print("\nExecuting query...")
            # Provide the DataFrame in the local scope for eval
            result = eval(code, globals(), {"df": self.df})
            display(result)

            return code, result
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize the assistant
    assistant = DataAnalysisAssistant('data/Flight Bookings.csv')

    # Load your data
    assistant.load_data('data/Flight Bookings.csv')

    # Perform initial analysis
    analysis_plan = assistant.analyze_data()

    # Query interface example
    code, result = assistant.query_data("which airline was most booked")
