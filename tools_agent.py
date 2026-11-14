import ollama


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return int(a) + int(b)


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return int(a) * int(b)


tools = {"add": add, "multiply": multiply}

question = "What is 1234 times 5678?"
messages = [{"role": "user", "content": question}]

response = ollama.chat(
    model="llama3.2",
    messages=messages,
    tools=[add, multiply],
)

for call in response.message.tool_calls or []:
    name = call.function.name
    args = call.function.arguments
    result = tools[name](**args)
    print(f"AI chose: {name}({args}) = {result}")

