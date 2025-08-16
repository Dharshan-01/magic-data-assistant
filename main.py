# main.py

import os
import io
import re
import json
import pandas as pd
import psycopg
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, List
from pydantic import BaseModel, Field

# --- Langchain Imports (for tool definition) ---
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()

# --- Initialize Google Gemini Client ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not found. Chat will not work.")
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error initializing Google Gemini client: {e}")

# Create the FastAPI application instance
app = FastAPI(
    title="Magic Data Assistant API",
    description="An API to chat with your data using a robust AI pipeline.",
    version="4.1.0" # Final version
)

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    table_name: str
    question: str

# --- Helper Functions (Unchanged) ---
def pandas_to_sql_type(dtype):
    if "int" in dtype.name: return "BIGINT"
    elif "float" in dtype.name: return "FLOAT"
    elif "datetime" in dtype.name: return "TIMESTAMP"
    else: return "TEXT"

def get_table_context(db_url: str, table_name: str):
    """Gets the table schema and date range for context."""
    conn = None
    try:
        conn = psycopg.connect(db_url)
        with conn.cursor() as cur:
            # Get schema
            schema_query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"
            cur.execute(schema_query)
            schema_rows = cur.fetchall()
            schema_str = ", ".join([f'"{col[0]}" ({col[1]})' for col in schema_rows])
            
            # Check for a 'Date' column to get the date range
            column_names = [col[0] for col in schema_rows]
            date_range_str = ""
            if "Date" in column_names:
                try:
                    date_range_query = f'SELECT MIN("Date"), MAX("Date") FROM "{table_name}";'
                    cur.execute(date_range_query)
                    min_date, max_date = cur.fetchone()
                    if min_date and max_date:
                        date_range_str = f"The data in this table spans from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}."
                except Exception:
                    # If date range query fails, just ignore it
                    pass

            return schema_str, date_range_str
    finally:
        if conn:
            conn.close()

# --- Tool Definition (using LangChain for schema generation) ---
@tool
def run_sql_query(sql_query: str = Field(description="The PostgreSQL query to be executed.")):
    """A tool to execute a SQL query against the database."""
    # This is just a definition for the AI, the actual execution happens below.
    pass

# Initialize the LLM with the tool definition
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools([run_sql_query])


# --- API Endpoints ---

@app.get("/")
async def read_root():
    return FileResponse('index.html')

# === FINAL ROBUST VERSION: Chat Endpoint with a Manual Two-Step AI Call ===
@app.post("/chat/")
async def chat_with_data(request: ChatRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured.")

    try:
        db_url = os.getenv("NEON_CONNECTION_STRING")
        if not db_url:
            raise ValueError("NEON_CONNECTION_STRING not found in .env file")
            
        table_schema, date_range = get_table_context(db_url, request.table_name)

        # --- STEP 1: Generate the SQL Query ---
        prompt_for_sql = f"""
        You are a PostgreSQL expert. Your only task is to generate a single, valid SQL query to answer the user's question about the given table, and then call the `run_sql_query` tool with that query.

        Table Name: "{request.table_name}"
        Schema: {table_schema}
        Context: {date_range} The current date is {pd.Timestamp.now().strftime('%Y-%m-%d')}.
        User Question: "{request.question}"

        Follow these rules precisely:
        1.  Use the provided Context to understand time-related questions. For example, "last week" refers to the last week within the data's date range, not the current date.
        2.  Generate a single PostgreSQL query.
        3.  CRITICAL: You MUST enclose all table and column identifiers in double quotes. For example: `SELECT "Weekly_Sales" FROM "walmart_sales"`. This is mandatory to preserve case-sensitivity.
        4.  Call the `run_sql_query` tool with the generated query.
        """
        
        ai_response = llm_with_tools.invoke(prompt_for_sql)
        
        if not ai_response.tool_calls:
            return {"answer": ai_response.content, "sql_query": "N/A", "query_result": []}

        sql_query = ai_response.tool_calls[0]["args"]["sql_query"]

        # --- STEP 2: Execute the SQL Query ---
        conn = None
        tool_result_str = ""
        query_data = []
        try:
            conn = psycopg.connect(db_url)
            with conn.cursor() as cur:
                cur.execute(sql_query)
                query_upper = sql_query.strip().upper()
                
                if query_upper.startswith("SELECT"):
                    columns = [desc[0] for desc in cur.description]
                    results = [dict(zip(columns, row)) for row in cur.fetchall()]
                    query_data = results
                    tool_result_str = f"Query returned {len(results)} rows. Data: {json.dumps(results, default=str)}" # Added default=str for date handling
                else:
                    rows_affected = cur.rowcount
                    conn.commit()
                    tool_result_str = f"Query executed successfully. {rows_affected} rows were affected."
        except Exception as e:
            print(f"SQL Execution Error: {e}")
            tool_result_str = f"Error executing SQL: {e}"
        finally:
            if conn:
                conn.close()

        # --- STEP 3: Generate the Final Answer ---
        prompt_for_answer = f"""
        You are a helpful data assistant. A user asked the following question: "{request.question}".
        To answer it, the following SQL query was run: `{sql_query}`.
        The result of the query was: "{tool_result_str}".

        Based on this information, provide a final, friendly answer to the user in plain English.
        """
        
        final_response = llm.invoke(prompt_for_answer)
        
        return {
            "answer": final_response.content,
            "sql_query": sql_query,
            "query_result": query_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# ... (The other endpoints: /upload-csv/, etc. remain unchanged) ...
@app.post("/upload-csv/")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), parse_dates=True)
        # FIX: Corrected the bad character range in the regular expression
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(file.filename)[0]).lower()
        conn = None
        try:
            db_url = os.getenv("NEON_CONNECTION_STRING")
            conn = psycopg.connect(db_url)
            with conn.cursor() as cur:
                columns_with_types = [f'"{col}" {pandas_to_sql_type(dtype)}' for col, dtype in df.dtypes.items()]
                create_query = (
                    f'DROP TABLE IF EXISTS "{table_name}";\n'
                    f"CREATE TABLE \"{table_name}\" (id SERIAL PRIMARY KEY, {', '.join(columns_with_types)});"
                )
                cur.execute(create_query)
                output = io.StringIO()
                df.to_csv(output, sep='\t', header=False, index=False)
                output.seek(0)
                column_names_for_copy = ", ".join([f'"{col}"' for col in df.columns])
                with cur.copy(f"COPY \"{table_name}\" ({column_names_for_copy}) FROM STDIN") as copy:
                    while data := output.read(1024):
                        copy.write(data)
                conn.commit()
            return {"message": f"Successfully created table '{table_name}' and inserted {len(df)} rows.", "database_table_created": table_name}
        except Exception as e:
            if conn: conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            if conn: conn.close()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error processing CSV file: {e}")