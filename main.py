import glob
from pprint import pprint as pp
import pdb
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from utils.agent import create_pandas_dataframe_agent
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from IPython.display import display
import scipy
from scipy import stats
import autopep8


class DataAnalysis(BaseModel):
    analysis_steps: List[str] = Field(description="Steps to analyze the data")
    cleaning_steps: Optional[List[str]] = Field(description="Steps to clean the data if necessary")
    pandas_code: str = Field(description="Pandas code to perform the analysis")


class DataAnalysisAssistant:
    def __init__(self, llm_model="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=llm_model)
        self.df = None
        self.original_df = None
        self.metadata = None
        self.max_retries = 3

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

    def _get_analysis_prompt(self, context: dict) -> PromptTemplate:
        """
        Generate a contextual prompt template for data analysis

        Args:
            context (dict): Contains:
                - df_info: DataFrame information including columns, dtypes, shape
                - metadata: Additional metadata about the dataset
                - previous_error: Any previous error message (optional)
                - attempt: Current attempt number (optional)
        """
        base_template = """
        You are a data scientist analyzing a dataset. Based on the following information:

        DATASET INFORMATION:
        {df_info}

        ADDITIONAL METADATA:
        {metadata}

        {error_context}

        Provide a detailed analysis plan that:
        1. Identifies meaningful columns for analysis
        - Skip ID columns, timestamps unless relevant for time series
        - Focus on numerical and categorical columns with analytical value
        - Consider relationships between related columns

        2. Suggests data cleaning steps that address:
        - Missing values
        - Outliers
        - Data type conversions
        - Format standardization
        - Any issues identified in the data info

        3. Provides pandas code that:
        - Implements the cleaning steps
        - Performs initial exploratory analysis
        - Creates meaningful aggregations and statistics
        - Identifies patterns and trends
        - Do not contain any comments in the code
        - {code_requirements}

        Your response must be in this exact JSON format:
        {{
            "analysis_steps": [
                "Detailed step-by-step analysis plan",
                "Each step should be specific and actionable"
            ],
            "cleaning_steps": [
                "Specific cleaning operations",
                "Include rationale for each cleaning step"
            ],
            "pandas_code": "Complete, executable pandas code that follows the requirements"
        }}

        CODE REQUIREMENTS:
        - Use 'df' as the dataframe name
        - Store final results in 'analysis_result'
        - Code must be complete and self-contained
        - Do not include import statements
        - Handle potential errors gracefully
        """

        code_requirements = "Produces clear, actionable insights"
        if context.get("attempt", 1) > 1:
            code_requirements += """
            - Address the previous error
            - Include additional error handling
            - Add data validation checks
            """

        return PromptTemplate(
            input_variables=["df_info", "metadata", "error_context", "code_requirements"],
            template=base_template
        )

    def _execute_code(self, code: str) -> any:
        """Safely execute code and return the result"""

        namespace = {
            "df": self.df,
            "scipy": scipy,
            "stats": stats,
            "pd": pd,
            "np": np,
            "analysis_result": None
        }

        code.replace("```", "").replace("python", "")
        exec(code, namespace)
        return namespace.get("analysis_result")

    def analyze_data(self):
        """
        Perform initial data analysis with AI-powered error correction
        """
        if self.df is None:
            raise ValueError("Please load data first using load_data()")

        parser = PydanticOutputParser(pydantic_object=DataAnalysis)
        attempts = 0
        last_error = None
        prev_code = None

        while attempts < self.max_retries:
            try:
                error_context = ""
                if last_error:
                    error_context = f"""
                    PREVIOUS RESPONSE:
                    {str(prev_code)}

                    PREVIOUS ERROR:
                    The last attempt failed with error: {str(last_error)}
                    Please modify the code to address this specific error.

                    ATTEMPT:
                    This is attempt {attempts + 1} of {self.max_retries}.
                    """

                code_requirements = """
                    - Produces clear, actionable insights
                    - use only pandas
                """
                if attempts > 0:
                    code_requirements += """
                    - Address the previous error
                    - Include additional error handling
                    - Add data validation checks
                    """

                context = {
                    "df_info": self._get_dataframe_info(),
                    "metadata": str(self.metadata),
                    "error_context": error_context,
                    "code_requirements": code_requirements
                }

                prompt = self._get_analysis_prompt(context)

                # Execute the chain
                chain = prompt | self.llm | parser
                response = chain.invoke(context)

                # Execute cleaning steps if provided
                if response.cleaning_steps:
                    print("\nExecuting cleaning steps:")
                    for step in response.cleaning_steps:
                        print(f"- {step}")

                print("\nExecuting analysis:")
                options = {
                    'aggressive': 1,
                    'max_line_length': 88,
                    'experimental': True
                }
                code = autopep8.fix_code(response.pandas_code, options=options)
                result = self._execute_code(code)

                display(result)
                return response

            except Exception as e:
                last_error = e
                prev_code = code if 'code' in locals() else None
                attempts += 1

                if attempts == self.max_retries:
                    print(f"Failed after {attempts} attempts. Last error: {str(e)}")
                    raise
                print(f"Attempt {attempts} failed: {e}")
                print("Requesting AI to fix the error...")

    def query_data(self, question):

        dfs = []
        data_dir = "data"
        for csv_file in glob.glob(f'{data_dir}/*.csv'):
            df = pd.read_csv(csv_file)
            dfs.append(df)

        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            dfs,
            verbose=True,
            allow_dangerous_code=True
        )  # note to self: https://python.langchain.com/v0.1/docs/integrations/toolkits/pandas/#multi-dataframe-example

        return agent.invoke(question)

    def _get_dataframe_info(self) -> str:
        """Get relevant DataFrame information for context"""
        info = {
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "shape": self.df.shape,
            "sample": self.df.head(3).to_dict()
        }
        return str(info)

    def debug(self):
        pdb.set_trace()


# Example usage
if __name__ == "__main__":
    # Initialize the assistant
    assistant = DataAnalysisAssistant()

    # Load your data
    assistant.load_data('data/Flight Bookings.csv')

    # Perform initial analysis
    analysis_plan = assistant.analyze_data()

    while True:
        question = input("> ")
        if question == "e":
            break
        pp(assistant.query_data(question))
