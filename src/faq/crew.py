from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# from crewai_tools import QdrantVectorSearchTool
import os
from dotenv import load_dotenv
from faq.tools.qdrant_vector_search_tool import (
    QdrantVectorSearchTool,
)

load_dotenv()


def embedding_function(text: str) -> list[float]:
    import openai

    openai_client = openai.Client(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


@CrewBase
class FaqCrew:
    """Faq crew"""

    vector_search_tool = QdrantVectorSearchTool(
        collection_name="contracts_business_5",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        custom_embedding_fn=embedding_function,
    )



    @agent
    def data_retrieval_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["data_retrieval_analysis_specialist"],  # type: ignore
            tools=[self.vector_search_tool],
        )

    @agent
    def source_citer_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["source_citer_specialist"],  # type: ignore
            tools=[self.vector_search_tool],
        )
    
    @agent
    def website_scraping_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["website_scraping_specialist"],
            tools=[self.vector_search_tool],
        )


    @agent
    def report_generation_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generation_specialist"],  # type: ignore
            tools=[self.vector_search_tool],
        )

    @task
    def retrieve_contracts_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_contracts_task"],  # type: ignore
        )

    @task
    def source_citer_task(self) -> Task:
        return Task(
            config=self.tasks_config["source_citer_task"],  # type: ignore
        )

    @task
    def scrape_website_task(self) -> Task:
        return Task(
            config=self.tasks_config["scrape_website_task"],
        )


    @task
    def generate_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_report_task"],  # type: ignore
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Faq crew"""
        return Crew(
            agents=self.agents,  # type: ignore
            tasks=self.tasks,  # type: ignore
            process=Process.sequential,
            verbose=True,
        )
