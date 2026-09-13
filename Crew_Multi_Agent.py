from crewai import Agent,Task,Crew,Process
from crewai.llm import LLM

llm=LLM(model="ollama/llama3")

researcher=Agent(role="researcher",
                 goal='Uncover groundbreaking technologies and trnds in. {topic}',
                 backstory=("you distill the most relevant and copelling insights from your vast knowledge base.""you identify key trends ,explain why they matter,a nd keep things concise."),
                 verbose=True,
                 allow_delegation=False,
                 llm=llm)

writer=Agent(role='Tech Content Strategist',
             goal='Create a compelling and insightful blog post on {topic}',
             backstory=("you are a skilled content strategist with a keen eye for detail and a passion for storytelling.""you transform complex technical concepts into compelling narratives that resonate with your audience."),
             verbose=True,
             allow_delegation=False,
             llm=llm)

research_task=Task( description=("Identify the top 3-5 ,ost significiant trends in {topic} this year""For each trend,provide 2-3 bullet points with key facts or implications""Your final answer Must be a concise ,well-structured list"),
                   expected_output="A bulleted list summary of the top 3-5 trends in specified topic",agent=researcher)
writer_task=Task(description=("Create a compelling and insightful blog post on {topic}""Incorporate the trends identified by the researcher, providing context and analysis""Your final answer must be a well-structured blog post with an engaging introduction, body, and conclusion"),
                 expected_output="A ~500-word blog post formatted in Markdown" ,agent=writer, context=[research_task])

crew=Crew(
    agents=[researcher,writer],
    tasks=[research_task,writer_task],
    process=Process.sequential,
    verbose=True
)

if __name__=="__main__":
    topic="Artificial Intelligence in Healthcare"
    result = crew.kickoff(inputs={"topic": topic})

    print("\n\n######")
    print("## Here is the final result:")
    print("#######")
    print(result)