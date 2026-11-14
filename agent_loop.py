import ollama


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return int(a) + int(b)


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return int(a) * int(b)


tools = {"add": add, "multiply": multiply}

question = "What is 1234 times 5678, plus 99? Use the tools step by step."
messages = [
    {"role": "system", "content": "You must use the tools for EVERY calculation. Never do math in your head. Pass only plain numbers to tools."},
    {"role": "user", "content": question},
]

for step in range(5):
    response = ollama.chat(
        model="llama3.2",
        messages=messages,
        tools=[add, multiply],
    )
    messages.append(response.message)

    if not response.message.tool_calls:
        print("Final answer:", response.message.content)
        break

    for call in response.message.tool_calls:
        name = call.function.name
        args = call.function.arguments
        try:
            result = tools[name](**args)
        except Exception as error:
            result = f"Error: {error}. Call the tool again using plain numbers only."
        print(f"Step {step + 1}: {name}({args}) = {result}")
        messages.append({"role": "tool", "content": str(result), "tool_name": name})
