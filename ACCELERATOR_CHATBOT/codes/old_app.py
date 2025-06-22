
# --- LLM Initialization ---
llm = LlamaCpp(
    model_path=config.MODEL_PATH,
    n_gpu_layers=-1,
    temperature=0.0,
    n_ctx=131072, # Increased context for potentially better performance with LlamaCpp
    verbose=True
)

# --- Database Engine Initialization ---
engine = create_engine(config.CONNECTION_STRING)


def generate_response(prompt: str) -> str:
    """
    Generates a response to a given prompt using the LangChain SQL Agent.

    Args:
        prompt: The user's query about the database.

    Returns:
        The generated response from the SQL agent.
    """
    db = SQLDatabase(
        engine,
        # Only include tables relevant to your queries.
        # This reduces the schema context given to the LLM, potentially improving accuracy.
        include_tables=['BeamParameters', 'MagnetControls', 'VacuumSystem', 'OperationalEvents', 'AcceleratorDevices']
    )

    # Create the SQL Database Toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create the SQL Agent
    # AgentType.ZERO_SHOT_REACT_DESCRIPTION is often a good choice for SQL agents
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Invoke the agent with the prompt
    response_obj = agent_executor.invoke({"input": prompt})

    return response_obj['output']

if __name__ == "__main__":
    # prompt = "What is the temperature of Quadrupole-1?"
    prompt = input("Enter your query about the accelerator database: (or type 'exit' to quit)\n")
    if prompt.lower() == 'exit':
        print("Exiting the application.")
        exit()
    try:
        response = generate_response(prompt)
        print("\n--- Final Response ---")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check the agent's verbose output above for more details.")