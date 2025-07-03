import asyncio

import asyncio

from agno.agent import Agent
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.qdrant import Qdrant
import os

#COLLECTION_NAME = "website-content-crewai-agno"
COLLECTION_NAME = "contracts_business_5"
 
vector_db = Qdrant(
    collection=COLLECTION_NAME,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
# Create a knowledge base with the seed URLs
knowledge_base = WebsiteKnowledgeBase(
    urls=["https://bocs-primature.sn"],
    # Number of links to follow from the seed URLs
    max_links=5,
    # Table name: ai.website_documents
    vector_db=vector_db,
)

# Create an agent with the knowledge base
agent = Agent(knowledge=knowledge_base, search_knowledge=True, debug_mode=True)

if __name__ == "__main__":
    # Comment out after first run
    asyncio.run(knowledge_base.aload(recreate=False))

    # Create and use the agent
    asyncio.run(agent.aprint_response("Que signifie BOCS?", markdown=True))