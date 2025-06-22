'''
Prompt for fire-detection & context analysis
'''

prompt = f"""
You are analyzing an infrared (thermal) video frame taken inside an industrial tunnel where a fire is occurring. The environment may or may not contain people. 

Please carefully observe the frame and provide insights about:
- Any signs of fire, smoke, or heat anomalies.
- Whether there appear to be people or human shapes present.
- Safety-related observations or risks associated with the fire and any detected people.
- Any other notable objects or situations related to fire safety in this industrial tunnel environment.

Focus your response on safety implications and situational awareness based on what is visible in the infrared image.

Response must not be more than 3-4 sentences.

Response:
"""

