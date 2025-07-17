import argparse

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama


# This function takes the name from chain_one and makes input for chain_two
def to_menu_input(restaurant_name: str) -> dict:
    return {"restaurant_name": restaurant_name}


if __name__ == "__main__":
    # Init LLM
    llm = ChatOllama(model="llama3.1")  # You can use any other Ollama-supported model

    title_prompt = PromptTemplate.from_template(
        """
        I want to open a restaurant for {cuisine} food. Suggest your top fency name for this.
        Return only one name.
        """
    )
    menu_prompt = PromptTemplate.from_template(
        """
        Suggest some menu items for {restaurant_name}
        """
    )

    chain_one = title_prompt | llm
    chain_two = to_menu_input | menu_prompt | llm

    # create argepares
    parser = argparse.ArgumentParser(description="URL of Webpage to be summarised")

    # add arguements for parser
    parser.add_argument("--cuisine", type=str, required=True, help="Type of cuisine")

    args = parser.parse_args()

    full_chain: Runnable = chain_one | chain_two

    output = full_chain.invoke({"cuisine": args.cuisine})

    print(output.content)
