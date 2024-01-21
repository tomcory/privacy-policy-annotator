from openai import OpenAI
from dotenv import load_dotenv
import tiktoken


def is_policy(html: str) -> (str, float):
    # load the OPENAI_API_KEY environment variable from the .env file and create an OpenAI client with it
    load_dotenv()
    client = OpenAI()

    # gpt-3.5-turbo uses the cl100k_base encoding, so we can use it to see how many tokens are used by our API call
    encoding = tiktoken.get_encoding('cl100k_base')

    # the system message tells GPT what to do
    system_msg = "Your role is to analyze the first few lines of a simplified HTML document and determine if the " \
                 "excerpt is likely part of a privacy policy. You should respond with only one word: 'true' if the " \
                 "excerpt seems to be from a privacy policy, 'false' if it likely is not, and 'unknown' if " \
                 "there's not enough information to decide. You should not provide any additional explanations or " \
                 "context in your response."
    system_len = len(encoding.encode(system_msg))

    # the first 100 tokens of the HTML string should be enough to determine whether the text is a privacy policy
    encoded_html = encoding.encode(html)
    encoded_excerpt = encoded_html[:100]
    user_msg = encoding.decode(encoded_excerpt)
    user_len = len(encoded_excerpt)

    # configure and query GPT
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # gpt-3.5-turbo performs very well across multiple languages and is cheap
        n=1,  # only generate one choice
        max_tokens=1,  # generate just one single token ('true', 'false' and 'unknown' each have 1 token, so they fit)
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    # extract the output text from the response message
    output = completion.choices[0].message.content
    output_len = len(encoding.encode(output))

    # calculate the cost of the API call based on the total number of tokens used
    cost = (system_len + user_len) * 0.0001 + output_len * 0.0002

    return output, cost
