# generate_factual_description.py

"""
Generates a long, FACTUAL description of an image by interrogating it
with a Visual Question Answering (VQA) model and synthesizing the answers.

1.  INTERROGATE (VQA):  A VQA model answers a list of predefined questions
                        about the image's content, extracting factual details.
2.  SYNTHESIZE (LLM):   A Large Language Model combines these Q&A facts into
                        a single, cohesive, high-fidelity paragraph.
3.  DECOMPRESS (Diffusion): The factual description is used to reconstruct the image.
"""
import torch
import requests
from PIL import Image
import logging

from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# This process is memory-intensive. Using 4-bit loading is highly recommended.
USE_4BIT = True

# --- Model IDs ---
VQA_MODEL = "Salesforce/blip-vqa-base"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
TEXT2IMG_MODEL = "runwayml/stable-diffusion-v1-5"


def main():
    # --- Load a sample image ---
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        logger.info(f"Downloading sample image from: {image_url}")
        original_image = Image.open(requests.get(image_url, stream=True).raw).convert(
            "RGB"
        )
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        return

    # ===================================================================
    # STEP 1: Interrogate the Image with VQA
    # ===================================================================
    logger.info("\n--- Step 1: Interrogating image with VQA model ---")
    try:
        vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL)
        vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL).to(DEVICE)
    except Exception as e:
        logger.error(f"Could not load VQA model: {e}")
        return

    # Define a list of questions to probe the image details
    questions = [
        "What is the main subject of this image?",
        "Describe the colors of the main subject.",
        "What is in the background?",
        "What is in the foreground?",
        "Describe the object on the left side of the image.",
        "Describe the object on the right side of the image.",
        "What is the lighting like in this scene?",
        "Are there any animals in this picture? If so, what kind?",
        "What textures are visible in this image?",
        "Is this scene indoors or outdoors?",
    ]

    extracted_facts = []
    logger.info("Asking questions about the image...")
    for question in questions:
        inputs = vqa_processor(original_image, question, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = vqa_model.generate(**inputs, max_new_tokens=50)
        answer = vqa_processor.decode(out[0], skip_special_tokens=True)
        extracted_facts.append(f"Q: {question}\nA: {answer}")
        logger.info(f" -> Q: {question} -> A: {answer}")

    facts_string = "\n\n".join(extracted_facts)

    # ===================================================================
    # STEP 2: Synthesize Facts into a Cohesive Description
    # ===================================================================
    logger.info("\n--- Step 2: Synthesizing facts with an LLM ---")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, torch_dtype=torch.float16, load_in_4bit=USE_4BIT
        )
    except Exception as e:
        logger.error(f"Could not load LLM. Error: {e}")
        return

    # A prompt that instructs the LLM to only use the provided facts
    synthesis_prompt = f"""
    [INST] You are a technical writer. Your task is to combine the following series of questions and answers about an image into a single, cohesive, descriptive paragraph. Do not add any new information, speculation, or creative details. Only use the facts provided in the Q&A list. Combine them smoothly into a factual summary.

    Here are the facts:
    {facts_string}

    Now, provide the synthesized, factual description of the image. [/INST]
    """

    messages = [{"role": "user", "content": synthesis_prompt}]
    tokenized_chat = llm_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)

    logger.info("Generating final high-fidelity description...")
    with torch.no_grad():
        outputs = llm_model.generate(
            tokenized_chat,
            max_new_tokens=512,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    factual_description = llm_tokenizer.decode(
        outputs[0][tokenized_chat.shape[-1] :], skip_special_tokens=True
    ).strip()

    text_filename = "factual_description.txt"
    with open(text_filename, "w", encoding="utf-8") as f:
        f.write(factual_description)
    logger.info(f"✅ High-fidelity description saved to '{text_filename}'")

    # ===================================================================
    # STEP 3: Decompress Text back to Image and Plot
    # ===================================================================
    logger.info("\n--- Step 3: Decompressing text to image and plotting ---")
    try:
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            TEXT2IMG_MODEL, torch_dtype=torch.float16
        ).to(DEVICE)
    except Exception as e:
        logger.error(f"Could not load diffusion model: {e}")
        return

    reconstructed_image = text2img_pipe(factual_description).images[0]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(reconstructed_image)
    axes[1].set_title("Reconstructed Image", fontsize=14)
    axes[1].axis("off")

    plot_filename = "factual_comparison.png"
    plt.savefig(plot_filename)
    logger.info(f"✅ Comparison plot saved to '{plot_filename}'")
    logger.info("\n--- Test Finished ---")


if __name__ == "__main__":
    main()
