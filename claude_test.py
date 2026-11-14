import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=300,
    messages=[{"role": "user", "content": "อธิบายว่า Docker คืออะไร แบบสั้นๆ"}],
)

print(response.content[0].text)
print("---")
print("Tokens used:", response.usage.input_tokens, "in +", response.usage.output_tokens, "out")
