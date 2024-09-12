import language_tool_python

# Initialize the language tool for English
tool = language_tool_python.LanguageTool('en-US')

def correct_text(input_seq):
    # Check and correct grammar
    matches = tool.check(input_seq)
    corrected_text = language_tool_python.utils.correct(input_seq, matches)
    return corrected_text

# Example usage
if _name_ == "_main_":
    while True:
        input_seq = input("Enter text: ")
        corrected_text = correct_text(input_seq)
        print(f"Corrected Text: {corrected_text}")