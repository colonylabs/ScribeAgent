import json
from typing import Any
from collections import defaultdict
from beartype import beartype
from agent import Agent
import re
from openai import OpenAI


class GPTAgent:
    """prompt-based agent that emits action given the history"""

    def __init__(self, prompt_dict, note, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.note = note
        
        self.reflect_prompt = prompt_dict["reflect"]
        self.map_prompt = prompt_dict["map"]
        self.stop_prompt = prompt_dict["stop"]
        self.intent_prompt = prompt_dict["intent"]

    
    def reflect(self, prompt, goal):
        print("\n" + "-"*15, "CALLING USER AGENT FOR", goal.upper(), "-"*15)

        if goal == "stop":
            sys_prompt = self.stop_prompt
        elif goal == "reflect":
            sys_prompt = self.reflect_prompt
        elif goal == "intent":
            sys_prompt = self.intent_prompt
        else:
            sys_prompt = self.map_prompt

        messages = [
                {"role": "system", "content": "You are a helpful assistant designed to solve web-based tasks"},
                {"role": "user", "content": (sys_prompt + prompt)},
            ]

        if goal == "stop" and self.note != "":
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to solve web-based tasks"},
                {"role": "user", "content": sys_prompt[:re.search("You will decide whether the task", sys_prompt).start()] + "Here're the notes from previous steps: " + self.note + "\n" + sys_prompt[re.search("You will decide whether the task", sys_prompt).start():] + prompt},
            ]

        print("[PROMPT]", messages[-1]["content"])
        
        response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

        generated_text = response.choices[0].message.content
        num_total_tokens = response.usage.total_tokens
        print("\n[RESPONSE]", generated_text)
        flag, action = self.analyze(generated_text, goal)
        print("\n[RESULTS]", flag, action)

        return flag, action


    def analyze(self, generated_text, goal):
        if goal == "stop":
            if generated_text.rfind("Summary:"):
                generated_text = generated_text[generated_text.rfind("Summary:"):].replace("Summary:\n", "Summary: ").replace("Summary: \n", "Summary: ")
            can_stop = "completed" in generated_text.lower() and "incomplete" not in generated_text.lower()
            generated_text = re.sub(r"\s+", " ", generated_text.replace("Summary:", "").strip())
            
            if not can_stop:
                self.note += generated_text.replace("incomplete", "")
            else:
                if "n/a" in generated_text.lower():
                    generated_text = "n/a"
                else:
                    generated_text = generated_text.replace("completed,", "")

            return can_stop, generated_text
            
        elif goal == "reflect":
            digitstr = ""
            for g in generated_text.lower():
                if g.isdigit() or g == "-":
                    digitstr += g
                elif len(digitstr) > 0:
                    break
            if digitstr == "":
                return -1, []
            return int(digitstr), []

        elif goal == "intent":
            return True, generated_text.replace("Detailed Task Objective: ", "")

        else:
            if "stop" in generated_text.lower():
                return True, "stop"
            elif "go back" in generated_text.lower():
                return True, "go back"

            for w in generated_text.split():
                w = w.lower()
                if w == "click":
                    sidx = re.search("\\[", generated_text).start()
                    generated_text = generated_text[sidx:]
                    eidx = re.search("\\]", generated_text).end()
                    return True, "click " + generated_text[:eidx]
  
                elif w == "type":
                    sidx = re.search("\\[", generated_text).start()
                    generated_text = generated_text[sidx:]
                    eidx = re.search("\\]", generated_text).end()
                    astr = generated_text[:eidx]
                    generated_text = generated_text[eidx:]
                    eidx = re.search("\\]", generated_text).end()
                    return True, "type " + astr + generated_text[:eidx]

                elif w == "press":
                    sidx = re.search("\\[", generated_text).start()
                    generated_text = generated_text[sidx:]
                    eidx = re.search("\\]", generated_text).end()
                    return True, "press " + generated_text[:eidx]

            return False, "scroll [down]"
