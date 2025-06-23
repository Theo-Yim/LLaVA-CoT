from openai_image_query import OpenAIImageQueryClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

#OPENAI_MODEL = "gpt-4o"
OPENAI_MODEL = "o4-mini"
#OPENAI_MODEL = "gpt-4o-mini"

# Initialize client
client = OpenAIImageQueryClient(model=OPENAI_MODEL)

# Stage 1. Summary
response = client.query_with_images_from_folder(
    # text_query="Analyze the scene. For each description, always include noun phrase with participial phrase.",
    # text_query="I have an image that I want you to analyze. As a first step, briefly explain what steps you'll take to do so.",
    # text_query="briefly explain what steps you'll take to analyze the scene.",
    # text_query="briefly explain where to majorly look into to analyze the scene.",
    # text_query="I want to focus on some aspects to piece together the activities within the scene. Where should I focus? List up the only few major regions of interest with the short captioning.",
    text_query="I want to focus on some aspects to piece together the activities within the scene. Where should I focus? List up the only few major regions of interest with noun category name (like person) followed by semicolon and noun phrase with short participial phrase (like person with red shirt)",
    # text_query="Is this scene person behind the counter wearing dark clothing and scanning items. Only answer yes, maybe, or no",
    folder_path="./samples",
    image_names=["inanam-taipan.jpg"]
)
print(response) 
# # The image shows a security camera view of the interior of a convenience store. There are multiple people inside, some standing near the cashier counter. The store is stocked with various items including snacks, packaged goods, and beverages. One individual in the foreground is holding several items.

# # Stage 2. Captioning path 1
# response = client.query_with_images_from_folder(
#     text_query="One individual in the foreground is holding several items. Describe it.",
#     folder_path="./test_theo/samples",
#     image_names=["inanam-taipan-closeup2.jpg"]
# )
# print(response)

# # Stage 2. Captioning path 2
# response = client.query_with_images_from_folder(
#     text_query="One individual in the foreground is holding several items. Describe it.",
#     folder_path="./test_theo/samples",
#     image_names=["inanam-taipan-closeup2.jpg"]
# )
# print(response)

# # Stage 3. Reasoning path 2
# response = client.query_with_images_from_folder(
#     text_query="Previous Description: The store is stocked with various items including snacks, packaged goods, and beverages. One individual in the foreground is holding several items. The individual in the foreground is holding several canned items. They are wearing a red shirt, dark pants, and a cap, and have a black bag strapped across their chest. They also have on a face mask and glasses. Check if previous description is correct by providing a chain-of-thought, logical explanation of the scene. This should outline step-by-step reasoning.",
#     folder_path="./test_theo/samples",
#     image_names=["inanam-taipan-closeup1.jpg"]
# )
# print(response)

# prev_response = """To verify the previous description:

# 1. **Setting and Items**: The image shows a store interior with shelves stocked with various items. This includes packaged goods visible on the shelves, consistent with snacks, canned goods, or similar items. The presence of beverages or specific item types like snacks cannot be explicitly confirmed without detailed observation of the labels.

# 2. **Individual in Foreground**: 
#     - **Appearance**: The person is in focus, wearing a red shirt and dark pants, as described.
#     - **Accessories**: The individual is wearing a cap and a face mask and has a black bag strapped across their chest.
#     - **Items Held**: The person is clearly holding several items, which appear to be cans. This confirms part of the description regarding holding canned items.

# 3. **Additional Observations**:
#     - **Store Layout**: The store appears to have multiple aisles with neatly organized shelves.
#     - **Camera Angle**: The image is likely from a security camera, given the perspective and the presence of a time/date stamp in the lower right corner.

# Overall, the previous description accurately captures the main elements of the scene portrayed in the image."""

# # Stage 4. CONCLUSION
# response = client.query_with_images_from_folder(
#     text_query=f"{prev_response} In conclusion, state the final answer in a clear and direct format. It must match the correct answer exactly.",
#     folder_path="./test_theo/samples",
#     image_names=["inanam-taipan.jpg"]
# )
# print(response)

