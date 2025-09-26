import pandas as pd  # Import pandas library for handling CSV files
from transformers import pipeline  # Import pipeline from transformers for easy model usage
import huggingface_hub  # Import huggingface_hub for authentication

# Login to Hugging Face using the provided token to access models
huggingface_hub.login("hf_your_token_here")

def read_csv_headers(file_path):
    """
    Reads the headers from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list: List of header names.
    """
    df = pd.read_csv(file_path, nrows=0)  # Read only the header row from the CSV file without data rows
    return list(df.columns)  # Return the column names as a list

def generate_description(header, generator):
    """
    Generates a short descriptive text for a CSV header using the language model.

    Args:
        header (str): The header name.
        generator: The text generation pipeline.

    Returns:
        str: Generated description.
    """
    prompt = f"The header '{header}' in a CSV file likely means"  # Create a prompt string for the model
    result = generator(prompt, max_new_tokens=10, num_return_sequences=1, pad_token_id=50256)  # Generate text using the model with limited tokens
    generated = result[0]['generated_text']  # Extract the generated text from the result
    desc = generated[len(prompt):].strip()  # Remove the prompt from the generated text and strip whitespace
    # Take only the first few words to keep it short
    words = desc.split()[:5]  # Split into words and take the first 5
    desc = ' '.join(words)  # Join them back into a string
    return f"likely {desc}"  # Return the description prefixed with 'likely'

def output_results(descriptions, output_file='output.txt'):
    """
    Outputs the descriptions to console and a text file.

    Args:
        descriptions (dict): Dictionary of header to description.
        output_file (str): Path to the output text file.
    """
    # Output to console
    for header, desc in descriptions.items():  # Loop through each header-description pair
        print(f"{header} → {desc}")  # Print the header and description with an arrow
    
    # Output to file
    with open(output_file, 'w', encoding='utf-8') as f:  # Open the file for writing with UTF-8 encoding
        for header, desc in descriptions.items():  # Loop through each pair again
            f.write(f"{header} → {desc}\n")  # Write each line to the file

def main():
    """
    Main function to orchestrate the script execution.
    """
    file_path = 'data.csv'  # Define the path to the input CSV file
    # Load the text generation model from Hugging Face
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
    
    # Read headers from the CSV file
    headers = read_csv_headers(file_path)

    # Generate descriptions for each header
    descriptions = {}  # Initialize an empty dictionary to store results
    for header in headers:  # Loop through each header
        desc = generate_description(header, generator)  # Generate description for the header
        descriptions[header] = desc  # Store in the dictionary
    
    # Output the results
    output_results(descriptions)

if __name__ == "__main__":  # Check if the script is run directly
    main()  # Call the main function  to execute the script