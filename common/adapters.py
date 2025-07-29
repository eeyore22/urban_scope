# common/adapters.py

class DefaultAdapter:
    def post_process_inputs(self, processed: dict) -> dict:
        return processed
    
    def format_prompt(self, raw_prompt: str, qa: dict, use_cot: bool = False) -> str:
        # Use the already formatted prompt
        prompt = raw_prompt 
        if use_cot: 
            COT_PROMPT_PREFIX = "Please explain your reasoning step by step. Answer: "
            prompt = raw_prompt + COT_PROMPT_PREFIX

        return prompt

class LlavaAdapter(DefaultAdapter):
    def post_process_inputs(self, processed: dict) -> dict:
        # LLaVA wants half-precision pixel_values
        processed["pixel_values"] = processed["pixel_values"].half()
        return processed
    
    def format_prompt(self, raw_prompt: str, qa: dict, use_cot: bool = False) -> str:
        # let DefaultAdapter handle CoT-prefix + {.} substitution
        base = super().format_prompt(raw_prompt, qa, use_cot)
        return f"<image>\n### Human: {base}\n### Assistant:"
        

