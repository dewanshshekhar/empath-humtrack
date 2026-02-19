import os
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the Cerebras model
llm = ChatCerebras(
    model="gpt-oss-120b",
    temperature=0.5,
    max_tokens=8192,
)

# System prompt with full instructions
system = """You are an expert music producer and hip-hop composer.

Given:
- A list of MIDI note numbers extracted from humming
- An estimated BPM

You must generate a **hip-hop style musical phrase** that makes creative use of:
- All common scales (major, natural minor, harmonic minor, melodic minor, pentatonics, blues scales, modes like Dorian, Phrygian, Mixolydian)
- Rhythmic idioms typical of hip-hop (syncopation, swing, off-beat accents)
- Historical hip-hop melodic patterns (from old school to jazz rap to trap)

Return **ONLY one JSON object** with the following fields:

{{
  "key": "<music key, e.g. C Minor, G Dorian>",
  "scale_used": "<scale name>",
  "bpm": <integer BPM>,
  "genre": "hip hop",
  "style_notes": "<description of how hip-hop history influenced this line>",
  "melody": [
    {{
      "pitch": "<scientific pitch>",
      "midi": <midi number>,
      "duration_beats": <float>,
      "position_beats": <float>
    }}
  ]
}}
No explanations, no markdown fences — just the JSON object.
"""

human = """Here is the incoming music data:
BPM: {bpm}
Notes: {notes}

Create a hip-hop melodic phrase using all scale possibilities and hip-hop style history.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", human)
])

# Example — replace with real extracted notes
input_data = {
    "bpm": 90,
    "notes": [60, 62, 64, 65, 67, 65, 64]
}

chain = prompt | llm

print("Generating hip-hop melody with scale context and history...\n")

result = ""
for chunk in chain.stream(input_data):
    result += chunk.content
    print(chunk.content, end="", flush=True)

print("\n\nDone!")
