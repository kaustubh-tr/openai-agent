from literun import Agent, ChatOpenAI, Tool, ArgsSchema, ToolRuntime


# 1. Define a function that uses BOTH an LLM argument and the new ToolRuntime
# The 'user_id' comes from the LLM.
# The 'ctx' comes from the Agent runtime.
def get_user_data(user_id: str, ctx: ToolRuntime) -> str:
    # Access the attributes passed at runtime
    # Use getattr for safety if key might be missing
    db_conn = getattr(ctx, "db_connection", "DefaultDB")
    request_id = getattr(ctx, "request_id", "Unknown")
    print(f"Fetching data for User {user_id} using {db_conn} [Req: {request_id}]")
    return f"Fetching data for User {user_id} using {db_conn} [Req: {request_id}]"


# 2. Wrap it in a Tool
# Note: We ONLY define 'user_id' in args_schema.
# We do NOT include 'ctx' in the schema, so the LLM doesn't see it.
tool = Tool(
    name="get_user_data",
    description="Get user info by ID",
    func=get_user_data,
    args_schema=[
        ArgsSchema(name="user_id", type=str, description="The ID of the user")
    ],
)

# 3. Setup Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = Agent(llm=llm, system_prompt="You are a helpful assistant.", tools=[tool])

# 4. Invoke with Runtime Context
# This dictionary {"db_connection": "ProdDB", ...} will be injected
# into the 'ctx' parameter of get_user_data().
print("--- Agent Response ---")
response = agent.invoke(
    user_input="Can you get info for user 12345?",
    runtime_context={"db_connection": "Production-SQL-01", "request_id": "req-abc-999"},
)

print(response.final_output)
