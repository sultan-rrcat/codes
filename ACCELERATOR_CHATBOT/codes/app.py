import json
import time
import config
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

class SchemaRAGSystem:
    def __init__(self, schema_file: str):
        self.schema_data = self._load_schema(schema_file)
        self.vectorstore = self._create_vector_store()
        
    def _load_schema(self, file_path: str) -> Dict:
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _create_vector_store(self):
        # Create LangChain Documents from schema
        documents = []
        
        # Add table information
        for table in self.schema_data["tables"]:
            content = f"Table: {table['name']}\nDescription: {table['description']}\nColumns:\n"
            for col in table["columns"]:
                content += f"- {col['name']} ({col['type']}): {col['description']}"
                if "example_values" in col:
                    content += f" (e.g., {', '.join(map(str, col['example_values']))})"
                content += "\n"
            documents.append(Document(page_content=content, metadata={"type": "table", "table_name": table['name']}))
        
        # Add relationships
        if "relationships" in self.schema_data:
            content = "Table Relationships:\n"
            for rel in self.schema_data["relationships"]:
                content += (f"{rel['table1']} ↔ {rel['table2']}: {rel['relationship']}\n"
                           f"Join condition: {rel.get('join_condition', 'N/A')}\n\n")
            documents.append(Document(page_content=content, metadata={"type": "relationship"}))
        
        # Add business rules
        if "business_rules" in self.schema_data:
            content = "Business Rules:\n" + "\n".join(f"- {rule}" for rule in self.schema_data["business_rules"])
            documents.append(Document(page_content=content, metadata={"type": "business_rules"}))
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        return Chroma.from_documents(documents=splits, embedding=embeddings)
    
    def retrieve_relevant_schema(self, query: str, k: int = 3) -> List[str]:
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

class SQLQueryGenerator:
    def __init__(self, db_uri: str, schema_rag: SchemaRAGSystem):
        self.db = SQLDatabase.from_uri(db_uri)
        self.rag = schema_rag
        self.llm = LlamaCpp(
                    model_path=config.MODEL_PATH,
                    n_gpu_layers=-1,
                    temperature=0.0,
                    n_ctx=131072, # Increased context for potentially better performance with LlamaCpp
                    verbose=False
                )
        
        self.system_prompt = """You are an expert SQL developer for particle accelerator control systems.
        Use the following schema information to generate accurate SQL queries:
        
        {schema_context}
        
        Rules:
        1. Always use TOP/LIMIT for queries that might return large results
        2. Include timestamps in historical queries
        3. Use JOINs carefully based on the described relationships
        4. Never modify data - only SELECT queries are allowed
        5. Always include the Timestamp column when showing historical data"""
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Question: {question}\n\nGenerate a SQL query to answer this question:")
        ])
    
    def generate_query(self, question: str) -> str:
        # Retrieve relevant schema info
        schema_context = "\n\n".join(self.rag.retrieve_relevant_schema(question))
        
        # Generate query with context
        chain = self.prompt_template | self.llm | StrOutputParser()
        return chain.invoke({
            "schema_context": schema_context,
            "question": question
        })

class ResponseFormatter:
    @staticmethod
    def format_result(query: str, result: List[Dict], execution_time: float) -> str:
        if not result:
            return "No results found for your query."
        
        response = f"Query executed in {execution_time:.2f}s\n\n"
        
        if len(result) == 1:
            response += "Result:\n"
            for k, v in result[0].items():
                response += f"- {k}: {v}\n"
        else:
            response += f"Found {len(result)} records:\n"
            
            # Show column headers
            headers = list(result[0].keys())
            response += " | ".join(headers) + "\n"
            response += "-" * (sum(len(h) for h in headers) + 3 * len(headers)) + "\n"
            
            # Show first 5 rows
            for row in result[:5]:
                response += " | ".join(str(row[h]) for h in headers) + "\n"
            
            if len(result) > 5:
                response += f"\n(Showing 5 of {len(result)} records)"
        
        return response

def main():
    # Initialize components
    rag = SchemaRAGSystem(config.DATABASE_SCHEMA_FILE)
    query_gen = SQLQueryGenerator(config.CONNECTION_STRING, rag)
    formatter = ResponseFormatter()
    
    # Create SQL engine
    engine = create_engine(config.CONNECTION_STRING)
    
    # Example conversation loop
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ").strip()
        if question.lower() == 'quit':
            break
        
        try:
            # Step 1: Generate query
            print("\nGenerating SQL query...")
            sql_query = query_gen.generate_query(question)
            print(f"\nGenerated SQL:\n{sql_query}")
            
            # Step 2: Execute query
            with engine.connect() as conn:
                print("\nExecuting query...")
                start_time = time.time()
                result = conn.execute(text(sql_query))
                execution_time = time.time() - start_time
                rows = [dict(row) for row in result.mappings()]
            
            # Step 3: Format response
            print("\n" + formatter.format_result(sql_query, rows, execution_time))
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Please try rephrasing your question or ask something else.")

if __name__ == "__main__":
    main()