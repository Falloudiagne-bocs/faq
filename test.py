from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool

# Initialize the tool
# Initialize the tool with predefined parameters
scrape_tool = ScrapeWebsiteTool(website_url='https://bocs-primature.sn',)

# Define an agent that uses the tool
# Example of using the tool with an agent
web_agent = Agent(
    role="Web Agent expert",
    goal="Answer the question from websites",
    backstory="An expert in web who can extract the relevente content using the query.",
    tools=[scrape_tool],
    verbose=True,
)

# Create a task for the agent to extract specific elements
task = Task(
    description="""
    Extract all informations from the featured section on bocs-primature.sn.
    
    """,
    expected_output="the content of the featured section",
    agent=web_agent,  
)

# Run the task through a crew
crew = Crew(agents=[web_agent], tasks=[task])  
result = crew.kickoff()
