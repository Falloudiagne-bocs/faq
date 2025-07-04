from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

import os
from dotenv import load_dotenv
from crewai_tools import WebsiteSearchTool
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

    website_vector_tool = WebsiteSearchTool(website='https://bocs-primature.sn')

    @agent
    def data_retrieval_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["data_retrieval_analysis_specialist"],
            tools=[self.vector_search_tool],
        )

    @agent
    def website_info_extractor_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["website_info_extractor_specialist"],
            tools=[self.website_vector_tool],
        )

    @agent
    def doc_source_citer_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["doc_source_citer_specialist"],
            tools=[self.vector_search_tool],
        )

    @agent
    def website_source_citer_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["website_source_citer_specialist"],
            tools=[self.website_vector_tool],
        )

    @agent
    def report_generation_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generation_specialist"],
            tools=[self.vector_search_tool, self.website_vector_tool],
        )

    @task
    def retrieve_info_from_doc_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_info_from_doc_task"],
        )

    @task
    def extract_info_from_website_task(self) -> Task:
        return Task(
            config=self.tasks_config["extract_info_from_website_task"],
        )

    @task
    def source_doc_citer_task(self) -> Task:
        return Task(
            config=self.tasks_config["source_doc_citer_task"],
        )

    @task
    def cite_website_source_task(self) -> Task:
        return Task( 
            config=self.tasks_config["cite_website_source_task"],
        )

    @task
    def generate_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_report_task"],
        )

    @crew
    def crew(self, **kwargs) -> Crew:
        query = kwargs.get("query", "").lower()

        tasks = [self.retrieve_info_from_doc_task(), self.source_doc_citer_task()]

        if "site" in query or "web" in query or "bocs-primature.sn" in query:
            tasks.append(self.extract_info_from_website_task())
            tasks.append(self.website_source_citer_task())

        tasks.append(self.generate_report_task())

        return Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )
